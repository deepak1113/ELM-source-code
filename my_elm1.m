%Deepak R Nayak 27-12-16 % ELM source code%
function [output_wt,TrainingAccuracy,TestingAccuracy,correctClassifiedSamples_Testing]=my_elm1(training_data,testing_data, output_train,output_test,numberHidden)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% training_data-  Input training dataset (N1*d where N1=# of training samples, N2=# of testing samples)
%   
% output_train- Output training dataset (N1*c where c= # of clases, d= feature dimension)
%   
% testing_data- Input testing dataset (N2*d, where N2=# of testing samples)
%   
% output_test=Output testing dataset (N2*c)

% Outputs:
% output_wt- Output weights (beta)
% TrainingAccuracy- Training accuracy
% TestingAccuracy- Testing accuracy
% correctClassifiedSamples_Testing- correctly classified testing samples

% Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory and applications", Neurocomputing 70.1 (2006): 489-501.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_time_train=cputime;
d=size(training_data,2);
training_data(:,d+1)=1; % input bias added in last column
%% input weight and bias (last column) initialization
%inputWeightBias=rand(size(training_data,2),numberHidden)*0.0001; 
inputWeightBias=rand(size(training_data,2),numberHidden)*2-1; %a in [-1,1]
aa=inputWeightBias;
%a=rand(size(training_data,2),numberHidden)*2-1; %a in [-1,1]
H_final=[];
for i=1:size(training_data,1)
    h=inputWeightBias'*training_data(i,:)';
    H = 1 ./ (1 + exp(-h));% sigmoid trans func
    %H=h;
    %H=sin(h); % sine tf
    %H=radbas(h); % Radial basis function tf
    H_final=[H_final;H'];
    clear H;
   
end
output_wt=pinv(H_final)*output_train; % calculated output weight (beta)
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train  
%% Calculate the training accuracy
actual_output=H_final*output_wt; %the calculated output of the training data
correctClassifiedSamples_Training=0;
 for i = 1 : size(output_train, 1)
        [x, index_desired_label]=max(output_train(i,:));
        [x, index_actual_label]=max(actual_output(i,:));
        if index_actual_label==index_desired_label
            correctClassifiedSamples_Training=correctClassifiedSamples_Training+1;
        end
 end
    TrainingAccuracy=correctClassifiedSamples_Training/size(output_train,1)
%% testing part
start_time_test=cputime;
d1=size(testing_data,2);
testing_data(:,d1+1)=1; % input bias added in last column
H_final_test=[];
for i=1:size(testing_data,1)
    h_test=aa'*testing_data(i,:)';
    H_test = 1 ./ (1 + exp(-h_test));% sigmoid trans func
    %H_test=h_test;
    %H_test=sin(h_test); % sine tf
    %H_test=radbas(h_test); % Radial basis function tf
    H_final_test=[H_final_test;H_test'];
    clear H_test;   
end
%% Calculate the testing accuracy
actual_output_test=H_final_test*output_wt; %the calculated output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test 

correctClassifiedSamples_Testing=0;
 for i = 1 : size(output_test, 1)
        [x, index_desired_label]=max(output_test(i,:));
        [x, index_actual_label]=max(actual_output_test(i,:));
        if index_actual_label==index_desired_label
            correctClassifiedSamples_Testing=correctClassifiedSamples_Testing+1;
        end
 end
    TestingAccuracy=correctClassifiedSamples_Testing/size(output_test,1)
end

