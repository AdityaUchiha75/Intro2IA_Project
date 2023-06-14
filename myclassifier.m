% Team 7 name: Power Rangers
function S = myclassifier(I,Mdl)
  S=[];
  I1 = prepro_img(double(I)/255); % pre-processing
  coo_parts = {        % coordinates to the three segments
      85:220,100:185;
      85:220,180:265;
      85:220,245:330
      };
  for j=1:3            % looping over the three segments
      I11=I1(coo_parts{j,1},coo_parts{j,2}); % choosing segments
      pred = predict(Mdl,FeatExt(I11));   % predicting segments 
      S=[S, pred];    % adding results to output 
  end
end

function I=prepro_img(I11)
        I11=medfilt2(rgb2gray(I11),[5,5]);       % median filtering
        thresh = otsuthresh(imhist(I11));  
        %thresh = 0.5859; %replacing with the average threshold of the dataset to reduce computation time, just in case
        I11 = I11<(thresh*0.90);           % adjust Otsu thresholding a bit
        I11=imopen(I11,strel('square', 5));% reduce some negative effects 
        I = bwmorph(I11, 'clean');         % remove single pixels 
end

function F=FeatExt(I)

	F=[]; % Empty feature vector to add stuff to
	F=[F,transpose(I(:))]; %  returning the image intensity values as a vector

end