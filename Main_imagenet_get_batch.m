function [imo,labelo] = Main_imagenet_get_batch(images, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [256, 256]-128 ;%设置的输入影像大小
opts.depth=8;
% opts.border = [128, 128] ; %去除边界
opts.keepAspect = false ; %不需要拉伸
opts.numAugments = 1; %1 ;增强次数
opts.transformation = 'none'; 
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,opts.depth,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);
im={};
imt=[];
label=[];
% fetch is true if images is a list of filenames (instead of
% a cell array of images)
% fetch = numel(images) >= 1 && ischar(images{1}) ;
% 
% % prefetch is used to load images in a separate thread
% prefetch = fetch & opts.prefetch ;
% 
if opts.prefetch %获取单独的影像吧
%    vl_imreadjpeg(images, 'numThreads', opts.numThreads, 'prefetch') ;
   imo = [] ;
   labelo=[];
  return ;
end
% end
% if fetch
%   im = vl_imreadjpeg(images,'numThreads', opts.numThreads) ;
% else
%   im = images ; %影像名称
% end


tfs = [] ;
switch opts.transformation
  case 'none'
    tfs = [
      .5 ;
      .5 ;
       0 ] ;
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case 'stretch'
  otherwise
    error('Uknown transformations %s', opts.transformation) ;
end
[~,transformations] = sort(rand(size(tfs,2), numel(images)), 1) ;

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,opts.depth) ; %需要
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

%初始化影像
imo = zeros(opts.imageSize(1), opts.imageSize(2), opts.depth, ...
            numel(images)*opts.numAugments, 'single') ;  %需要
labelo = zeros(opts.imageSize(1), opts.imageSize(2), 1, ...
            numel(images)*opts.numAugments, 'single') ;  %需要


for i=1:numel(images)

  % acquire image
%     imt = imread(images{i}) ;
%     imt = single(imt) ; % faster than im2single (and multiplies by 255)
    f1=images{i}; %2017
    [d1,n1,ext]=fileparts(f1);
    [d2,~,~]=fileparts(d1);
    f2=[d2,'\img_2018\image_2018',n1(11:end),ext]; %2018
    f3=[d2,'\mask\mask_2017_2018',n1(11:end),ext];  %mask
    
    img1=imread(f1);
    img2=imread(f2);
    img3=imread(f3);
    
    %制作标签 change 1 nochange 2
    tmp=img3; tmp(img3==255)=1; tmp(img3~=255)=2;
    
    imt(:,:,:,i)=single(cat(3,img1,img2)); %叠加光谱
    label(:,:,1,i)= single(tmp); %类别标签为1 2； 从1 开始
    
    label=single(label);
    imt = single(imt) ; % faster than im2single (and multiplies by 255)
  
  %灰度影像 一个波段
  if size(imt,3) == 1
    imt = cat(3, imt, imt, imt) ;
%     label = cat(3, label, label, label) ;
  end
 

  % crop & flip
  w = size(imt,2) ;
  h = size(imt,1) ;
  %增强次数
    switch opts.transformation
      case 'stretch'
        sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
        dx = randi(w - sz(2) + 1, 1) ; %随机点位且位于内部
        dy = randi(h - sz(1) + 1, 1) ;
        flip = floor(rand(1)*3);%rand > 0.5 ;
        otherwise 
          %验证样本不进行裁剪 大小应该就是原来的
        tf = tfs(:, transformations(mod(1-1, numel(transformations)) + 1)) ;
        sz = opts.imageSize(1:2) ;
        dx = floor((w - sz(2)) * tf(2)) + 1 ;
        dy = floor((h - sz(1)) * tf(1)) + 1 ;
        flip = floor(tf(3)*3) ; %应该是 0
    end
    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    
%     if flip, sx = fliplr(sx) ; end  %水平
    %crop
    if flip <= 1 && flip>=0.0001
        sx = (fliplr(sx));
    elseif flip <= 2 && flip > 1
        sy = (fliplr(sy)); %其实是垂直翻转
    end

    imo(:,:,:,i) = imt(sy,sx,:,i) ; % 水平翻转
    labelo(:,:,1,i) = label(sy,sx,1,i) ; %

end

end
