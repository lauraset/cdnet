function net = Main_model(varargin)
% 修改为 resnet unet的结构， 输入影像为 128 x 128 x 8 个通道
% 前面的 resnet-50 保持不变，后面的结构改为对称版本，且加上短连接
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification
channel = 8;
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(channel, 1) ;
opts.colorDeviation = zeros(channel) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
lastAdded.depth = channel; %3 ;

%基本构成 上采样和下采样
function Conv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.relu = true ;
  args.downsample = false ;
  args.bias = false ;
  args = vl_argparse(args, varargin) ;
  if args.downsample, stride = 2 ; else stride = 1 ; end
 if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end 
  %其实用了 BN 就可以不用偏差项（COV)
  
  net.addLayer([name  '_conv'], ...
               dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
                          'stride', stride, ....
                          'pad', (ksize - 1) / 2, ...
                          'hasBias', args.bias, ...
                          'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
               lastAdded.var, ...
               [name '_conv'], ...
               pars) ;
  net.addLayer([name '_bn'], ...
               dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
               [name '_conv'], ...
               [name '_bn'], ...
               {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
  lastAdded.depth = depth ;
  lastAdded.var = [name '_bn'] ;
  if args.relu
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 lastAdded.var, ...
                 [name '_relu']) ;
    lastAdded.var = [name '_relu'] ;
  end
end

function DeConv(name, ksize, depth, varargin)
%首先连接
% Helper function to add a DeConvolutional + BatchNorm + ReLU
% sequence to the network.
%修改为上采样
  args.relu = true ;%默认激活
  args.upsample = false ;%默认不上采样
  args.bias = false ; %默认无偏差项
  args.copy=false; %增加连接层
  args = vl_argparse(args, varargin) ;
  if args.upsample, stride = 2 ; else stride = 1 ; end
  if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end

  %上采样层
  %输入输出层的顺序不一样
  %upsample 类似stride
  %crop  需要裁剪去多余的部分
  net.addLayer([name  '_deconv'], ...
               dagnn.ConvTranspose('size', [ksize ksize depth lastAdded.depth ], ...
                           'hasBias', args.bias, ...
                          'upsample', [stride, stride],...
                          'crop',[0, (ksize-1)/2,0, (ksize-1)/2],...
                          'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
               lastAdded.var, ...
               [name '_deconv'], ...
                pars) ;
  net.addLayer([name '_bn'], ...
               dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
               [name '_deconv'], ...
               [name '_bn'], ...
               {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
  lastAdded.depth = depth ;
  lastAdded.var = [name '_bn'] ;
  if args.relu
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 lastAdded.var, ...
                 [name '_relu']) ;
    lastAdded.var = [name '_relu'] ; %输出变量名始终是 relu
  end
end
% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1', 7, 64, ...
     'relu', true, ...
     'bias', true, ...  %增加了偏差项
     'downsample', true) ;  %大小减半

net.addLayer(...
  'conv1_pool' , ...
  dagnn.Pooling('poolSize', [3 3], ...
                'stride', 2, ...
                'pad', 1,  ...
                'method', 'max'), ...  %大小减半
  lastAdded.var, ...
  'conv1') ;
lastAdded.var = 'conv1' ;

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------

%% resnet-50结构: 选择前面五层
for s = 2:5 %for s = 2:5

  switch s
    case 2, sectionLen = 3 ;
    case 3, sectionLen = 4 ; % 8 ;
    case 4, sectionLen = 6 ; % 23 ; % 36 ;
    case 5, sectionLen = 3 ;
  end

  % -----------------------------------------------------------------------
  % Add intermediate segments for each section
  for l = 1:sectionLen
    depth = 2^(s+4) ;
    sectionInput = lastAdded ;
    name = sprintf('conv%d_%d', s, l)  ;

    % Optional adapter layer
    if l == 1
      Conv([name '_adapt_conv'], 1, 2^(s+6), 'downsample', s >= 3, 'relu', false) ;
    end
    sumInput = lastAdded ;

    % ABC: 1x1, 3x3, 1x1; downsample if first segment in section from
    % section 2 onwards.
    lastAdded = sectionInput ;
    %Conv([name 'a'], 1, 2^(s+4), 'downsample', (s >= 3) & l == 1) ;
    %Conv([name 'b'], 3, 2^(s+4)) ;
    Conv([name 'a'], 1, 2^(s+4)) ;
    Conv([name 'b'], 3, 2^(s+4), 'downsample', (s >= 3) & l == 1) ; %减小影像大小
    Conv([name 'c'], 1, 2^(s+6), 'relu', false) ;

    % Sum layer
    net.addLayer([name '_sum'] , ...
                 dagnn.Sum(), ...
                 {sumInput.var, lastAdded.var}, ...
                 [name '_sum']) ;
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 [name '_sum'], ...
                 name) ;
    lastAdded.var = name ;
  end
end

% -----------------------------------------------------------------------
%% unet 上采样回去
%% 1. U-NET上采样第一层
%第六层：与第五层是对称，block有3个
%直接将这层将为 2048-_1024,且上采样
  DeConv('conv6', 3, 1024, 'upsample',true) ; %上采样
  
%% 2. 连接变换中间层： 输入变为了 8 x 8 x 1024
%第七层到第九层：与第四层到第二层对称，采用resnet结构
for s = 7:9 %for s = 2:5
  switch s
    case 9, sectionLen = 3 ;id=['conv2_3'];
    case 8, sectionLen = 4 ;id=['conv3_4']; % 8 ;
    case 7, sectionLen = 6 ; id=['conv4_6'];% 23 ; % 36 ;
%     case 5, sectionLen = 3 ;
  end
  
  for l = 1:sectionLen
    depth = 2^(s-4) ;
    sectionInput = lastAdded ;
    name = sprintf('conv%d_%d', s, l)  ;

   
    %需要修改的地方：1.输入加入了前面的层 2.中间层上采样
    %拼接 对称位置的输入与上一层的输入
    if l ==1  %只有第一个结构才会连接，且仅增加通道数目
    concat = dagnn.Concat('dim', 3); % 深度,必不会增加深度
    net.addLayer([name, 'concat'], concat, {id, lastAdded.var }, {[name,'_concatx']}, {});
    lastAdded.var=[name,'_concatx']; %输入的变量名称变化了
    depth=  lastAdded.depth;%通道数增加
     lastAdded.depth=depth*2;
    %降低维度
        if s<= 8
            DeConv([name '_adapt_conv'], 3, 2^(11-s+6), 'upsample', true, 'relu', false) ;
        else
            Conv([name '_adapt_conv'], 1, 2^(11-s+6), 'relu', false) ;%降维
        end
    end
    sumInput = lastAdded ;
    lastAdded = sectionInput ; 
   
    Conv([name 'a'], 1,  2^(11-s+4)) ;
    if l==1 && s<=8
        DeConv([name 'b'], 3,  2^(11-s+4), 'upsample', true) ; %上采样 只有第一个
    else
        Conv([name 'b'], 3,  2^(11-s+4));
    end
    Conv([name 'c'], 1,  2^(11-s+6), 'relu', false) ;

    % Sum layer
    net.addLayer([name '_sum'] , ...
                 dagnn.Sum(), ...
                 {sumInput.var, lastAdded.var}, ...
                 [name '_sum']) ;
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 [name '_sum'], ...
                 name) ;
    lastAdded.var = name ;
  end
  
  %降通道
   Conv([name '_down'], 1,  2^(11-s+5),'relu', false);

end

%% 3. 上采样最后两层：现在得到了32 x 32 x 128
%% 最后2层
%  
% 拼接
    name='conv10';
    Conv([name '_down'], 1, 64,'relu', false); %维度降低
     concat = dagnn.Concat('dim', 3); % 深度
    net.addLayer([name, 'concat'], concat, {  'conv1', lastAdded.var }, {[name,'_concatx']}, {});
    lastAdded.var=[name,'_concatx']; %输入的变量名称变化了
    lastAdded.depth=lastAdded.depth*2;%通道数增加
% 上采样
    DeConv([name 'b'], 3, 64, 'upsample',true) ; %上采样
 %% 最后1层
% 拼接
    name='conv11';
    concat = dagnn.Concat('dim', 3); % 深度
    net.addLayer([name, 'concat'], concat, {  'conv1_relu', lastAdded.var }, {[name,'_concatx']}, {});
    lastAdded.var=[name,'_concatx']; %输入的变量名称变化了
    lastAdded.depth=lastAdded.depth*2;%通道数增加
% 上采样
    DeConv([name 'b'], 3, 64, 'upsample', true) ; %上采样
 
   %% 1 x1 卷积层: 输出类别个数为2
   pred = dagnn.Conv('size',[1,1,64,2], 'pad', 0, 'stride', 1, 'hasBias', true);
   net.addLayer('pred', pred, {lastAdded.var},{'prediction'},{'pred_f1','pred_b1'});

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

% net.addLayer('top1error', ...
%              dagnn.Loss('loss', 'classerror'), ...
%              {'prediction', 'label'}, ...
%              'top1error') ;
% 
% net.addLayer('top5error', ...
%              dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
%              {'prediction', 'label'}, ...
%              'top5error') ;

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [128 128 8] ;%修改了
% net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
% net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;

net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;

net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;
net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
net.meta.augmentation.jitterScale  = [0.4, 1.1] ;
%net.meta.augmentation.jitterSaturation = 0.4 ;
%net.meta.augmentation.jitterContrast = 0.4 ;

% net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

%lr = logspace(-1, -3, 60) ;
lr = [0.1 * ones(1,30), 0.01*ones(1,30), 0.001*ones(1,30)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numSubBatches = 4 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

% Init parameters randomly
net.initParams() ;

% For uniformity with the other ImageNet networks, t
% the input data is *not* normalized to have unit standard deviation,
% whereas this is enforced by batch normalization deeper down.
% The ImageNet standard deviation (for each of R, G, and B) is about 60, so
% we adjust the weights and learing rate accordingly in the first layer.
%
% This simple change improves performance almost +1% top 1 error.
p = net.getParamIndex('conv1_f') ;
net.params(p).value = net.params(p).value / 100 ;
net.params(p).learningRate = net.params(p).learningRate / 100^2 ;

for l = 1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.BatchNorm')
    k = net.getParamIndex(net.layers(l).params{3}) ;
    net.params(k).learningRate = 0.3 ;
  end
end

end
