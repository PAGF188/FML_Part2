% compute Haralick features of the images and apply the 1-NN classifier
clear all;
% images path
pathImaxes='images';
numImaxes=864; % number of images

% neighbours for different distances 
offset1 = [0 1; -1 1; -1 0; -1 -1]; % d=1
offset2 = [0 2; -1 2; -2 1; -2 0; -2 -1; -1 -2]; % d=2
offset3 = [0 3; -1 3; -2 2; -3 1; -3 0; -3 -1;-2 -2; -1 -3]; % d=3
offset4 = [0 4; -1 4; -2 4; -2 3; -3 3; -3 2; -4 2: -4 1; -4 0; -4 -1; -4 -2; -3 -2; -3 -3; -2 -3; -2 -4; -1 -4]; % d=4 

for i=1:numImaxes;
    f=[];
    filename = sprintf('%s/images/%06d.bmp', pathImaxes, i-1);
    rgb = imread(filename);
    grey = rgb2gray(rgb);
    % Coocurrence matrix for distance d=1
    glcm=graycomatrix(grey, 'offset', offset1, 'Symmetric', true);
    fs=graycoprops(glcm,{'contrast','homogeneity', 'correlation', 'Energy'});
    f=[f mean(fs.Contrast) mean(fs.Correlation) mean(fs.Energy) mean(fs.Homogeneity)];
    % Coocurrence matrix for distance d=2
    glcm=graycomatrix(grey, 'offset', offset2, 'Symmetric', true);
    fs=graycoprops(glcm,{'contrast','homogeneity', 'correlation', 'Energy'});
    f=[f mean(fs.Contrast) mean(fs.Correlation) mean(fs.Energy) mean(fs.Homogeneity)];
    % Coocurrence matrix for distance d=3
    glcm=graycomatrix(grey, 'offset', offset3, 'Symmetric', true);
    fs=graycoprops(glcm,{'contrast','homogeneity', 'correlation', 'Energy'});
    f=[f mean(fs.Contrast) mean(fs.Correlation) mean(fs.Energy) mean(fs.Homogeneity)];
    features(i,:)=f;
end

% read picture ID of training and test samples, and read class ID of
% training and test samples
trainTxt = sprintf('%s/train.txt', pathImaxes)
testTxt = sprintf('%s/test.txt', pathImaxes)
[trainIDs, trainClassIDs] = ReadOutexTxt(trainTxt);
[testIDs, testClassIDs] = ReadOutexTxt(testTxt);

 % classification test 
    trains=features(trainIDs', :);
    tests=features(testIDs', :);    
    trainNum = size(trains,1);
    testNum = size(tests,1);

% use L1 distance as metric measure
    [final_accu,PreLabel] = NNClassifierL1(trains',tests',trainClassIDs,testClassIDs);
    accu_list3 = final_accu;
    close all;