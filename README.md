# ELM source code
Inputs:

training_data-  Input training dataset (N1*d where N1=# of training samples, N2=# of testing samples)
  
output_train- Output training dataset (N1*c where c= # of clases, d= feature dimension)
  
testing_data- Input testing dataset (N2*d, where N2=# of testing samples)
  
output_test=Output testing dataset (N2*c)

Outputs:

output_wt- Output weights (beta)

TrainingAccuracy- Training accuracy

TestingAccuracy- Testing accuracy

correctClassifiedSamples_Testing- correctly classified testing samples
