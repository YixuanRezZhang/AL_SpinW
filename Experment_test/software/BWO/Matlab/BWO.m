%白鲸优化算法
function [fvalbest, xposbest,Curve] = BWO(Npop,Max_it,lb,ub,nD,fobj)
% disp('Beluga Whale Optimization is optimizing your problem');
fit = inf*ones(Npop,1);
newfit = fit;
Curve = inf*ones(1,Max_it);
kk = zeros(1,Max_it);
Counts_run = 0;
if size(ub,2)==1
    lb = lb*ones(1,nD); ub = ub*ones(1,nD);
end

%% 初始化
pos = rand(Npop,nD).*(ub-lb)+lb;
for i = 1:Npop
    fit(i,1) = fobj(pos(i,:));
    Counts_run = Counts_run+1;
end
[fvalbest,index]=min(fit);
xposbest = pos(index,:);
T = 1;

%% 迭代
while T <= Max_it
    newpos = pos;
    WF = 0.1-0.05*(T/Max_it);  % 鲸落概率
    kk = (1-0.5*T/Max_it)*rand(Npop,1); % 平衡因子
    for i = 1:Npop
        if kk(i) > 0.5 % exploration phase
            %% 勘探-游泳
            r1 = rand(); r2 = rand();
            RJ = ceil(Npop*rand);   % Roulette Wheel Selection
            while RJ == i
                RJ = ceil(Npop*rand);
            end
            if nD <= Npop/5
                params = randperm(nD,2);
                newpos(i,params(1)) = pos(i,params(1))+(pos(RJ,params(1))-pos(i,params(2)))*(r1+1)*sin(r2*2*pi);
                newpos(i,params(2)) = pos(i,params(2))+(pos(RJ,params(1))-pos(i,params(2)))*(r1+1)*cos(r2*2*pi);
            else
                params=randperm(nD);
                for j = 1:floor(nD/2)
                    newpos(i,2*j-1) = pos(i,params(2*j-1))+(pos(RJ,params(1))-pos(i,params(2*j-1)))*(r1+1)*sin(r2*2*pi);
                    newpos(i,2*j) = pos(i,params(2*j))+(pos(RJ,params(1))-pos(i,params(2*j)))*(r1+1)*cos(r2*2*pi);
                end
            end
        else  % exploitation phase
            %% 开发-捕食
            r3 = rand(); r4 = rand(); C1 = 2*r4*(1-T/Max_it);
            RJ = ceil(Npop*rand);   % Roulette Wheel Selection
            while RJ == i
                RJ = ceil(Npop*rand);
            end
            alpha=3/2;
            sigma=(gamma(1+alpha)*sin(pi*alpha/2)/(gamma((1+alpha)/2)*alpha*2^((alpha-1)/2)))^(1/alpha); % Levy flight
            u=randn(1,nD).*sigma;
            v=randn(1,nD);
            S=u./abs(v).^(1/alpha);
            KD = 0.05;
            LevyFlight=KD.*S;
            newpos(i,:) = r3*xposbest - r4*pos(i,:) + C1*LevyFlight.*(pos(RJ,:)-pos(i,:));
        end
        % boundary
        Flag4ub = newpos(i,:)>ub;
        Flag4lb = newpos(i,:)<lb;
        newpos(i,:)=(newpos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        newfit(i,1) = fobj(newpos(i,:));    % fitness calculation
        Counts_run = Counts_run+1;
        if newfit(i,1) < fit(i,1)
            pos(i,:) = newpos(i,:);
            fit(i,1) = newfit(i,1);
        end
    end
    
    for i = 1:Npop
        %% 鲸落
        if kk(i) <= WF
            RJ = ceil(Npop*rand); r5 = rand(); r6 = rand(); r7 = rand();
            C2 = 2*Npop*WF;
            stepsize2 = r7*(ub-lb)*exp(-C2*T/Max_it);
            newpos(i,:) = (r5*pos(i,:) - r6*pos(RJ,:)) + stepsize2;
            % boundary
            Flag4ub = newpos(i,:)>ub;
            Flag4lb = newpos(i,:)<lb;
            newpos(i,:)=(newpos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
            newfit(i,1) = fobj(newpos(i,:));    % fitness calculation
            Counts_run = Counts_run+1;
            if newfit(i,1) < fit(i,1)
                pos(i,:) = newpos(i,:);
                fit(i,1) = newfit(i,1);
            end
        end
    end
    [fval,index]=min(fit);
    if fval<fvalbest
        fvalbest = fval;
        xposbest = pos(index,:);
    end
    
    kk_Record(T) = kk(1);
    Curve(T) = fvalbest;
    T = T+1;
end
% display(['The function call is ', num2str(Counts_run)]);