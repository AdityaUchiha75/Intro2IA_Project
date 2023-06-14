%team 7- Code developed to train the model

% read all files
dd = dir('Train/*.png');
t=tic;
fileNames = {dd.name}; 

Ims = cell(numel(fileNames),1);
read = csvread('Train/labels.txt');

for ii = 1:numel(fileNames)    
   Ims{ii,1} = fileNames{ii};
   Ims{ii,2} = double(imread(append('Train/', fileNames{ii})))/255;
   Ims{ii,3} = read(ii,2);
   Ims{ii,4} = read(ii,3);
   Ims{ii,5} = read(ii,4);
end
 
tot_size=size(Ims,1);


%get_train_labels
clear labels
j=1;
for i=1:tot_size
    labels(j,:)=Ims{i,3};
    labels(j+1,:)=Ims{i,4};
    labels(j+2,:)=Ims{i,5};
    j=j+3;
end


coo_parts = {
    85:220,100:185;
    85:220,180:265;
    85:220,245:330
    };

fprintf('Extracting training features...\n');


k=1;
clear patterns
for i=1:tot_size
	I=preprocess_img1(Ims{i,2}); % Get image i from the training data
    for j=1:3
        I11 = I(coo_parts{j,1},coo_parts{j,2});
        patterns(k,:)=FeatureExtraction(I11); % Extract features
        k=k+1;
    end
end


total_size=size(labels,1);

train_size=0.7*total_size; % 70% split for train
val_size=0.3*total_size; % 30% split for validation

tr_label=labels(1:train_size,:);
v_label=labels(train_size+1:total_size,:);

tr_patterns=patterns(1:train_size,:);
v_patterns=patterns(train_size+1:total_size,:);

fprintf('Building model...\n');



% class trees - AdaBoost
%k=6;
%Mdl = fitcensemble(tr_patterns,tr_label); 

% tr = templateTree('MaxNumSplits',15); 
% Mdl = fitcensemble(tr_patterns,tr_label, 'Learners',tr);

Mdl = fitcknn(patterns,labels); %fitting over the entire train set
Mdl.NumNeighbors=4;

save("model.mat",'Mdl')


fprintf('\nResubstitution error: %5.2f%%\n\n',100*resubLoss(Mdl));

% If ensemble, then view the first descision tree; click on the nodes to display data about them
if isa(Mdl,'classreg.learning.classif.ClassificationEnsemble')
	view(Mdl.Trained{1},'Mode','graph');
end


fprintf('Predicting validation set...\n');

validation_pred = predict(Mdl,v_patterns);
toc(t);

accuracy = mean(validation_pred == v_label);
fprintf('Validation accuracy: %5.2f%%\n',accuracy*100);

f=figure(2);
if (f.Position(3)<800)
	set(f,'Position',get(f,'Position').*[1,1,1.5,1.5]); %Enlarge figure
end
confusionchart(v_label, validation_pred, 'ColumnSummary','column-normalized', 'RowSummary','row-normalized');
title(sprintf('Validation accuracy: %5.2f%%\n',accuracy*100));
