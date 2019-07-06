function Main_train(net, exp,varargin)
%FNCTRAIN Train FCN model using MatConvNet

 run matconvnet-1.0-beta25/matlab/vl_setupnn ;
 addpath(genpath('matconvnet-1.0-beta25\matlab'));

% experiment and data paths ����·��
path='E:\yinxcao\taskcd\code\data\';
opts.expDir = [path,exp] ;
opts.dataDir = [path,'image'] ; %Ӱ��λ�� trian\img_2017
opts.modelType = 'resnet' ;
% opts.sourceModelPath =[ path,'models\imagenet-resnet-50-dag.mat'] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup  ���ݼ�׼��
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat') ;
opts.numFetchThreads = 12 ; % not used yet
opts.lite = false ;

% training options (SGD)
opts.train = struct() ;
opts.train.gpus = [1]; %�����GPU�����ţ�һ�㲻��0����1
opts.train.batchSize = 20 ;
opts.train.numSubBatches = 10 ;
opts.train.learningRate = 1e-2 * [ones(1,10), 0.1*ones(1,5)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
%resnet
% net=Main_model();

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 12 segmentation dataset plus Berkeley's additional
% segmentations
% ׼������
if exist(opts.imdbPath,'file')
  imdb= load(opts.imdbPath) ;
else
  imdb = Main_image_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ; %���ú�����ȡ���ݼ�
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set = imdb.images.set;

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;


% % ��ѵ�����ľ�ֵ  ����ֵ

% imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
% if exist(imageStatsPath)
%   load(imageStatsPath, 'averageImage') ;
% else
%     %�����Լ������ľ�ֵ
% %     averageImage = getImageStats(opts, net.meta, imdb) ;
% %     save(imageStatsPath, 'averageImage') ;
% end
% % % ���µľ�ֵ�ı��ֵ
% net.meta.normalization.averageImage = averageImage;

% Get training and test/validation subsets ��ȡ���ݼ�
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
                 
[net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;

 % -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
%�������
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

net_ = net.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
% bopts.imageSize = meta.normalization.imageSize ;
% bopts.border = meta.normalization.border ;
% bopts.averageImage = []; 
%  bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;

fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
% �ж϶�������Ϊѵ�����ǲ���
for i = 1:length(batch)
    if imdb.images.set(batch(i)) == 1 %1Ϊѵ�������ļ���
        images(i) = strcat([imdb.imageDir.train filesep] , imdb.images.name(batch(i)));
    else
        images(i) = strcat([imdb.imageDir.test filesep] , imdb.images.name(batch(i)));
    end
end
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
% Ӱ���һ������: ���Ի���ѵ������Ҫ
vmax=imdb.images.data_max;
vmin=imdb.images.data_min;

if ~isVal
  % training  �ü�+��ת ����Ӱ��ͱ�ǩ������
  [im, labels] = Main_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
%   [im,labels] = Main_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0,...                          ..
%                               'transformation', 'none') ;%����ǿ����
   im=(im-vmin)./(vmax-vmin)+vmin;%��һ��
   % ��Ӱ������� ��С������ λ�� 0��0.01����
    fraction = rand(1)./100;
    noise = randn(size(im)); %
    noise = fraction.*(noise+abs(min(noise(:))));
    im = (im + noise)./(max(im(:) + noise(:)));%�������ٹ�һ��
else
  % validation: disable data augmentation
  [im, labels]  = Main_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
   im=(im-vmin)./(vmax-vmin)+vmin;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
    labels = gpuArray(labels) ;
  end
%   labels = imdb.images.label(batch) ;
    inputs = {'input', im, 'label', labels} ;
end

% ��ѵ�������ľ�ֵ ���ڹ�һ��
% -------------------------------------------------------------------------
function averageImage = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ; %ѵ�������ĸ���
batch = 1:length(train);
fn = getBatchFn(opts, meta) ;
train = train(1: 100: end);   %ÿһ�ٸ����ֵ
avg = {};
for i = 1:length(train)-2 %��ֹ���ݼ���������������
    temp = fn(imdb, batch(train(i):train(i)+99)) ;
    temp = temp{2};  %���� im����
    avg{end+1} = mean(temp, 4) ; %ÿ��patch�ľ�ֵ
end

averageImage = mean(cat(4,avg{:}),4) ;  %ƽ��Ӱ��
% ��GPU��ʽ��ת��Ϊcpu��ʽ�ı����������������GPU��
averageImage = gather(averageImage);



