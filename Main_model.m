function net = Main_model(varargin)
% �޸�Ϊ resnet unet�Ľṹ�� ����Ӱ��Ϊ 128 x 128 x 8 ��ͨ��
% ǰ��� resnet-50 ���ֲ��䣬����Ľṹ��Ϊ�Գư汾���Ҽ��϶�����
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

%�������� �ϲ������²���
function Conv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.relu = true ;
  args.downsample = false ;
  args.bias = false ;
  args = vl_argparse(args, varargin) ;
  if args.downsample, stride = 2 ; else stride = 1 ; end
 if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end 
  %��ʵ���� BN �Ϳ��Բ���ƫ���COV)
  
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
%��������
% Helper function to add a DeConvolutional + BatchNorm + ReLU
% sequence to the network.
%�޸�Ϊ�ϲ���
  args.relu = true ;%Ĭ�ϼ���
  args.upsample = false ;%Ĭ�ϲ��ϲ���
  args.bias = false ; %Ĭ����ƫ����
  args.copy=false; %�������Ӳ�
  args = vl_argparse(args, varargin) ;
  if args.upsample, stride = 2 ; else stride = 1 ; end
  if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end

  %�ϲ�����
  %����������˳��һ��
  %upsample ����stride
  %crop  ��Ҫ�ü�ȥ����Ĳ���
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
    lastAdded.var = [name '_relu'] ; %���������ʼ���� relu
  end
end
% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1', 7, 64, ...
     'relu', true, ...
     'bias', true, ...  %������ƫ����
     'downsample', true) ;  %��С����

net.addLayer(...
  'conv1_pool' , ...
  dagnn.Pooling('poolSize', [3 3], ...
                'stride', 2, ...
                'pad', 1,  ...
                'method', 'max'), ...  %��С����
  lastAdded.var, ...
  'conv1') ;
lastAdded.var = 'conv1' ;

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------

%% resnet-50�ṹ: ѡ��ǰ�����
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
    Conv([name 'b'], 3, 2^(s+4), 'downsample', (s >= 3) & l == 1) ; %��СӰ���С
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
%% unet �ϲ�����ȥ
%% 1. U-NET�ϲ�����һ��
%�����㣺�������ǶԳƣ�block��3��
%ֱ�ӽ���㽫Ϊ 2048-_1024,���ϲ���
  DeConv('conv6', 3, 1024, 'upsample',true) ; %�ϲ���
  
%% 2. ���ӱ任�м�㣺 �����Ϊ�� 8 x 8 x 1024
%���߲㵽�ھŲ㣺����Ĳ㵽�ڶ���Գƣ�����resnet�ṹ
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

   
    %��Ҫ�޸ĵĵط���1.���������ǰ��Ĳ� 2.�м���ϲ���
    %ƴ�� �Գ�λ�õ���������һ�������
    if l ==1  %ֻ�е�һ���ṹ�Ż����ӣ��ҽ�����ͨ����Ŀ
    concat = dagnn.Concat('dim', 3); % ���,�ز����������
    net.addLayer([name, 'concat'], concat, {id, lastAdded.var }, {[name,'_concatx']}, {});
    lastAdded.var=[name,'_concatx']; %����ı������Ʊ仯��
    depth=  lastAdded.depth;%ͨ��������
     lastAdded.depth=depth*2;
    %����ά��
        if s<= 8
            DeConv([name '_adapt_conv'], 3, 2^(11-s+6), 'upsample', true, 'relu', false) ;
        else
            Conv([name '_adapt_conv'], 1, 2^(11-s+6), 'relu', false) ;%��ά
        end
    end
    sumInput = lastAdded ;
    lastAdded = sectionInput ; 
   
    Conv([name 'a'], 1,  2^(11-s+4)) ;
    if l==1 && s<=8
        DeConv([name 'b'], 3,  2^(11-s+4), 'upsample', true) ; %�ϲ��� ֻ�е�һ��
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
  
  %��ͨ��
   Conv([name '_down'], 1,  2^(11-s+5),'relu', false);

end

%% 3. �ϲ���������㣺���ڵõ���32 x 32 x 128
%% ���2��
%  
% ƴ��
    name='conv10';
    Conv([name '_down'], 1, 64,'relu', false); %ά�Ƚ���
     concat = dagnn.Concat('dim', 3); % ���
    net.addLayer([name, 'concat'], concat, {  'conv1', lastAdded.var }, {[name,'_concatx']}, {});
    lastAdded.var=[name,'_concatx']; %����ı������Ʊ仯��
    lastAdded.depth=lastAdded.depth*2;%ͨ��������
% �ϲ���
    DeConv([name 'b'], 3, 64, 'upsample',true) ; %�ϲ���
 %% ���1��
% ƴ��
    name='conv11';
    concat = dagnn.Concat('dim', 3); % ���
    net.addLayer([name, 'concat'], concat, {  'conv1_relu', lastAdded.var }, {[name,'_concatx']}, {});
    lastAdded.var=[name,'_concatx']; %����ı������Ʊ仯��
    lastAdded.depth=lastAdded.depth*2;%ͨ��������
% �ϲ���
    DeConv([name 'b'], 3, 64, 'upsample', true) ; %�ϲ���
 
   %% 1 x1 �����: ���������Ϊ2
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

net.meta.normalization.imageSize = [128 128 8] ;%�޸���
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
