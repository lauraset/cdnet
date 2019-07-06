function Main_train(net, exp,varargin)
%FNCTRAIN Train FCN model using MatConvNet

 run matconvnet-1.0-beta25/matlab/vl_setupnn ;
 addpath(genpath('matconvnet-1.0-beta25\matlab'));

% experiment and data paths 数据路径
path='E:\yinxcao\taskcd\code\data\';
opts.expDir = [path,exp] ;
opts.dataDir = [path,'image'] ; %影像位置 trian\img_2017
opts.modelType = 'resnet' ;
% opts.sourceModelPath =[ path,'models\imagenet-resnet-50-dag.mat'] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup  数据集准备
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat') ;
opts.numFetchThreads = 12 ; % not used yet
opts.lite = false ;

% training options (SGD)
opts.train = struct() ;
opts.train.gpus = [1]; %填的是GPU索引号，一般不是0就是1
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
% 准备数据
if exist(opts.imdbPath,'file')
  imdb= load(opts.imdbPath) ;
else
  imdb = Main_image_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ; %调用函数获取数据集
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set = imdb.images.set;

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;


% % 求训练集的均值  除均值

% imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
% if exist(imageStatsPath)
%   load(imageStatsPath, 'averageImage') ;
% else
%     %导入自己制作的均值
% %     averageImage = getImageStats(opts, net.meta, imdb) ;
% %     save(imageStatsPath, 'averageImage') ;
% end
% % % 用新的均值改变均值
% net.meta.normalization.averageImage = averageImage;

% Get training and test/validation subsets 获取数据集
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
%保存代码
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
% 判断读入数据为训练还是测试
for i = 1:length(batch)
    if imdb.images.set(batch(i)) == 1 %1为训练索引文件夹
        images(i) = strcat([imdb.imageDir.train filesep] , imdb.images.name(batch(i)));
    else
        images(i) = strcat([imdb.imageDir.test filesep] , imdb.images.name(batch(i)));
    end
end
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
% 影像归一化参数: 测试或者训练都需要
vmax=imdb.images.data_max;
vmin=imdb.images.data_min;

if ~isVal
  % training  裁剪+反转 （对影像和标签都做）
  [im, labels] = Main_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
%   [im,labels] = Main_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0,...                          ..
%                               'transformation', 'none') ;%不增强样本
   im=(im-vmin)./(vmax-vmin)+vmin;%归一化
   % 对影像加噪声 很小的噪声 位于 0，0.01区间
    fraction = rand(1)./100;
    noise = randn(size(im)); %
    noise = fraction.*(noise+abs(min(noise(:))));
    im = (im + noise)./(max(im(:) + noise(:)));%加噪声再归一化
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

% 求训练样本的均值 用于归一化
% -------------------------------------------------------------------------
function averageImage = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ; %训练样本的个数
batch = 1:length(train);
fn = getBatchFn(opts, meta) ;
train = train(1: 100: end);   %每一百个求均值
avg = {};
for i = 1:length(train)-2 %防止数据集不是整数倍出错
    temp = fn(imdb, batch(train(i):train(i)+99)) ;
    temp = temp{2};  %数据 im部分
    avg{end+1} = mean(temp, 4) ; %每个patch的均值
end

averageImage = mean(cat(4,avg{:}),4) ;  %平均影像
% 将GPU格式的转化为cpu格式的保存起来（如果有用GPU）
averageImage = gather(averageImage);



