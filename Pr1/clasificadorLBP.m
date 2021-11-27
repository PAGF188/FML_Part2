% apply the 1NN classifier using the LBP texture features
clear all;
% images path
pathImaxes='images';
numImaxes=864; % number of images
mapping=getmapping(8, 'riu2'); % mapping type: radius, number of neighbours and LBP type

% LBP feature computation
for i=1:numImaxes;
    filename = sprintf('%s/images/%06d.bmp', pathImaxes, i-1);
    rgb = imread(filename);
    grey = double(rgb2gray(rgb));
    features(i,:)=lbp(grey,1,8,mapping,'h');
end

% read picture ID of training and test samples, and read class ID of
% training and test samples
trainTxt = sprintf('%s/train.txt', pathImaxes);
testTxt = sprintf('%s/test.txt', pathImaxes);
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