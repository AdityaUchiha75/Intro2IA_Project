tic
% Team 7 name: Power Rangers
data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,(2:4));

my_labels = zeros(size(true_labels));
N = size(img_nrs);
load model.mat Mdl %knn_model: trained over the whole dataset
for n = 1:N
    k = img_nrs(n);
    im = imread(sprintf('Train/captcha_%04d.png', k));
    my_labels(k,:) = myclassifier(im,Mdl);
    %fprintf("Image %04d/1200 done\n",k);
end

fprintf('\n\nAccuracy: \n');
fprintf('%f\n\n',mean(sum(abs(true_labels - my_labels),2)==0));
toc

