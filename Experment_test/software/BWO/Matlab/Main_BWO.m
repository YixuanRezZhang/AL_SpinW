
% ----------------------- README ------------------------------------------
%     最后一次修改   ：2024/2/8
%                       欢迎关注₍^.^₎♡  
%     项目           ：白鲸优化算法BWO
%     内容来源文献   ：Zhong Changting, Li Gang, Meng Zeng. Beluga whale 
%        optimization: A novel nature-inspired metaheuristic algorithm [J]. 
%        Knowledge-Based Systems, 2022, 251.
%     微信公众号     ：KAU的云实验台(可咨询定制)  
%     CSDN/知乎      ：KAU的云实验台  
%     付费代码(更全) ：https://mbd.pub/o/author-a2iWlGtpZA==  
%     免费代码       ：公众号后台回复"资源" 
% -------------------------------------------------------------------------


%% 清理空间
close all
clear 
clc

%% 算法参数及测试函数
SearchAgents=100; 
Fun_name='F13';  
Max_iterations=500; 
[lowerbound,upperbound,dimension,fitness]=fun_info(Fun_name);

%% BWO
[Best_score,Best_pos,BWO_curve]=BWO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness);

%% 优化结果
figure('Position',[500 500 660 290])
%Draw search space
subplot(1,2,1);
fun_plot(Fun_name);
title('Parameter space')
xlabel('x_1');
ylabel('x_2');
zlabel([Fun_name,'( x_1 , x_2 )'])

%Draw objective space
subplot(1,2,2);
semilogy(BWO_curve,'Color','g');

title('Objective space')
xlabel('Iterations');
ylabel('Best score');

axis tight
grid on
box on

legend('BWO')

display(['The best optimal value of the objective function found by BWO is : ', num2str(Best_score)]);

        



