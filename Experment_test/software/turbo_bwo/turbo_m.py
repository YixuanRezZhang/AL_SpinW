###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math, os, sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
import random

from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from .gp import train_gp, GP
from .turbo_1 import Turbo1
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


class TurboM(Turbo1):
    """The TuRBO-m algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, np.array, shape (d,).
    ub : Upper variable bounds, np.array, shape (d,).
    n_init : Number of initial points *FOR EACH TRUST REGION* (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    n_trust_regions : Number of trust regions
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo5 = TurboM(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, n_trust_regions=5)
        turbo5.optimize()  # Run optimization
        X, fX = turbo5.X, turbo5.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        n_trust_regions,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        record_file=f'{os.getcwd()}/para_results.txt',
    ):
        self.n_trust_regions = n_trust_regions
        self.record_file=record_file
        super().__init__(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            min_cuda=min_cuda,
            device=device,
            dtype=dtype,
        )

        self.succtol = 3
        self.failtol = max(5, self.dim)

        # Very basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough trust regions to do initial evaluations"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)

    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next

    def load_gp(self, train_x, train_y, use_ard, X_cand, hypers={}):
        """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
        assert train_x.ndim == 2
        assert train_y.ndim == 1
        assert train_x.shape[0] == train_y.shape[0]
        assert train_x.min() >= 0.0 and train_x.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(train_y), train_y.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        train_y = (deepcopy(train_y) - mu) / sigma

        # Figure out what device we are running on
        if len(train_x) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype
        
        train_x = torch.tensor(train_x).to(device=device, dtype=dtype)
        train_y = torch.tensor(train_y).to(device=device, dtype=dtype)
    
        # Create hyper parameter bounds
        noise_constraint = Interval(5e-4, 0.2)
        if use_ard:
            lengthscale_constraint = Interval(0.005, 2.0)
        else:
            lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))  # [0.005, sqrt(dim)]
        outputscale_constraint = Interval(0.05, 20.0)
    
        # Create models
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
        ard_dims = train_x.shape[1] if use_ard else None
        model = GP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)
    
        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, model)
    
        # Initialize model hypers
        model.load_state_dict(hypers)

        # Switch to eval mode
        model.eval()
        likelihood.eval

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = model.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
    
        return y_cand

    

    def _select_candidates_each_TR(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        # X_next = np.zeros((self.batch_size, self.dim))
        # idx_next = np.zeros((self.batch_size, 1), dtype=int)

        X_all_next = []
        idx_all_next = []

        for i in range(self.n_trust_regions):
            X_next = np.zeros((self.batch_size, self.dim))
            idx_next = np.zeros((self.batch_size, 1), dtype=int)
            for k in range(self.batch_size):
                
                j = np.argmin(y_cand[i, :, k])
                assert y_cand[i, :, k].min() == y_cand[i, j, k]
                X_next[k, :] = deepcopy(X_cand[i, j, :])
                idx_next[k, 0] = i
                assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf
                # Make sure we never pick this point again
                y_cand[i, j, :] = np.inf
                
            X_all_next.append(X_next)
            idx_all_next.append(idx_next)
            
        X_next = np.vstack(X_all_next)
        idx_next = np.vstack(idx_all_next)

        return X_next, idx_next
    
    def _read_log(self):

        with open(self.record_file, 'r') as f:
            l = f.readlines()

        X = []
        Y = []

        for i in l:
            info = i.split()
            _x = np.array([float(num) for num in info[1].split('_') if num])
            X.append(_x)
            Y.append(np.array([float(info[2])]))

        order = np.argsort(np.squeeze(np.array(Y)))
        _X = np.array([X[i] for i in order])                
        #_Y = np.array([Y[i] for i in order])

        return _X
    
    def Levy(self):
        beta = 2  # (0~2)
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = 0.01 * np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        zz = np.power(np.absolute(v), (1 / beta))
        step = np.divide(u, zz)
        return step
        
    def optimize(self):
        """Run the full optimization process."""
        # Create initial points for each TR
        init = int(os.popen('cat '+self.record_file+' | wc -l').read())
        if init > 0:
            X_old = self._read_log()
        Tstep = 0
        MaxTstep = np.ceil(self.max_evals/(self.n_trust_regions*self.batch_size))
        for i in range(self.n_trust_regions):
            _X_init = latin_hypercube(self.n_init, self.dim)
            _X_init = from_unit_cube(_X_init, self.lb, self.ub)
            if init > 0 and i==0:
                X_init = np.vstack((_X_init, X_old))
            else:
                X_init = _X_init
            fX_init = np.array([[self.f(x)] for x in X_init])
            #fX_init = np.array(self.f(X_init))
            
            # Update budget and set as initial data for this TR
            self.X = np.vstack((self.X, X_init))
            self.fX = np.vstack((self.fX, fX_init))
            self.n_init = len(X_init)
            self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
            self.n_evals += self.n_init

            if self.verbose:
                fbest = fX_init.min()
                print(f"TR-{i} starting from: {fbest:.4}")
                sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            # Generate candidates from each TR
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                # Get the points, values the active values
                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)

                # Get the values from the standardized data
                fX = deepcopy(self.fX[idx, 0].ravel())

                # Don't retrain the model if the training data hasn't changed
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # Create new candidates
                X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                    X, fX, length=self.length[i], n_training_steps=n_training_steps, hypers=self.hypers[i]
                )

            X_cands = deepcopy(X_cand)
            y_cands = deepcopy(y_cand)
            ### Where to add the BWO position control
            kk = np.random.random(X_cands.shape[0])*(1 - 0.5 * (Tstep + 1) / MaxTstep)
            #print(f'kk: {kk}')
            for i in range(self.n_trust_regions):
                
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                # Get the points, values the active values
                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)

                # Get the values from the standardized data
                fX = deepcopy(self.fX[idx, 0].ravel())

                if fX.min() <= self.fX.min()*1.1:
                    X_cands[i] = X_cand[i]
                else:
                    if kk[i] > 0.5:
                        # Exploration phase - swimming
                        r1 = np.random.random_sample((self.n_cand))*self.length[i]/2
                        r2 = np.random.random_sample((self.n_cand))*self.length[i]/2
                        RJ = np.random.randint(self.n_trust_regions)
                        while RJ == i:
                            RJ = np.random.randint(self.n_trust_regions)
                        assert len(X_cand[i])==len(X_cand[RJ])
                        if self.dim <= self.n_trust_regions/5:
                            indices = np.arange(self.dim)
                            np.random.shuffle(indices)
                            
                            params = [indices[0], indices[1]]
                            X_cands[i][:, params[0]] = X_cand[i][:, params[0]] + (X_cand[RJ][:, params[0]] - X_cand[i][:, params[1]]) * (r1 + 1) * np.sin(r2 * 2 * math.pi)
                            X_cands[i][:, params[1]] = X_cand[i][:, params[1]] + (X_cand[RJ][:, params[0]] - X_cand[i][:, params[1]]) * (r1 + 1) * np.cos(r2 * 2 * math.pi)
                        else:
                            params = np.arange(self.dim)
                            np.random.shuffle(params)
                            for j in range(round(self.dim/2)):
                                X_cands[i][:, 2*j-1] = X_cand[i][:, params[2*j-1]] + (X_cand[RJ][:, params[0]] - X_cand[i][:, params[2*j-1]]) * (r1 + 1) * np.sin(r2 * 2 * math.pi)
                                X_cands[i][:, 2*j] = X_cand[i][:, params[2*j]] + (X_cand[RJ][:, params[0]] - X_cand[i][:, params[2*j]]) * (r1 + 1) * np.cos(r2 * 2 * math.pi)
                    else:
                        # Development Phase - Predation
                        r3 = np.random.random_sample((self.n_cand))*self.length[i]/2
                        r4 = np.random.random_sample((self.n_cand))*self.length[i]/2
                        r3 = np.expand_dims(r3,axis=-1)
                        r4 = np.expand_dims(r4,axis=-1)
                        C1 = 2*r4*(1-(Tstep + 1) / MaxTstep)
                        RJ = np.random.randint(self.n_trust_regions)
                        while RJ == i:
                            RJ = np.random.randint(self.n_trust_regions)
                        LevyFlight = self.Levy()
                        xposbest = self.X[np.argmin(self.fX)]
                        if xposbest.ndim!=X_cand[i].ndim:
                            xposbest = np.expand_dims(xposbest, axis=0)
                        #print(r3.shape, xposbest.shape, X_cand[i].shape, C1.shape, LevyFlight.shape)
                        X_cands[i] = r3 * np.tile(xposbest, (self.n_cand, 1)) - r4 * X_cand[i] + C1 * LevyFlight* (X_cand[RJ] - X_cand[i])

                X_cands = np.clip(X_cands, 0.0, 1.0)
                y_cands[i, :, :] = self.load_gp(train_x=X, train_y=fX, use_ard=self.use_ard, X_cand=X_cands[i], hypers=self.hypers[i])
                    
            Tstep += 1

            
            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cands, y_cands)         
            
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            X_next = from_unit_cube(X_next, self.lb, self.ub)

            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])
            #fX_next = np.array(self.f(X_next))

            # Update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}  # Remove model hypers
                    fX_i = fX_next[idx_i]

                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            #self.n_evals += self.batch_size
            self.n_evals += len(X_next)
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # Check if any TR needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:  # Restart trust region if converged
                    idx_i = self._idx[:, 0] == i

                    if self.verbose:
                        n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                        print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                        sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1  # Remove points from trust region
                    self.hypers[i] = {}  # Remove model hypers

                    # Create a new initial design
                    X_init = latin_hypercube(self.n_init, self.dim)
                    X_init = from_unit_cube(X_init, self.lb, self.ub)
                    fX_init = np.array([[self.f(x)] for x in X_init])
                    #fX_init = np.array(self.f(X_init))

                    # Print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                        sys.stdout.flush()

                    # Append data to local history
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init
