 
clc
% clear
 run E:\yinxcao\taskcd\code\matconvnet-1.0-beta25\matlab\vl_setupnn ;
addpath E:\yinxcao\taskcd\code\matconvnet-1.0-beta25\examples ;

% 导入model
net=load('E:\yinxcao\taskcd\code\data\exp_unet\net-epoch-10.mat');
net1=net.net;
net2 = dagnn.DagNN.loadobj(net1) ;

% net = cnn_imagenet_deploy() ;
% modelPath = fullfile(opts.expDir, 'net-deployed.mat');
% net_ = net.saveobj() ;
% save(modelPath, '-struct', 'net_') ;
% clear net_ ;
net2.mode = 'test' ;
% 导入准备数据
imdb = load('E:\yinxcao\taskcd\code\data\image\imdb.mat') ;
path='E:\yinxcao\taskcd\code\data\';
opts.expDir = [path,'exp'] ;
opts.dataDir = [path,'image'] ; %影像位置 trian\img_2017

% 找到训练与测试集
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;
% 影像归一化参数: 测试或者训练都需要
vmax=imdb.images.data_max;
vmin=imdb.images.data_min;

i=2;
    index = opts.train.train(i);
    % 读取测试的样本
   path=(fullfile(imdb.imageDir.train,imdb.images.name{index}));
    f1=path; %2017
    [d1,n1,ext]=fileparts(f1);
    [d2,~,~]=fileparts(d1);
    f2=[d2,'\img_2018\image_2018',n1(11:end),ext]; %2018
    f3=[d2,'\mask\mask_2017_2018',n1(11:end),ext];  %mask
    
    img1=imread(f1);img1=img1(:,:,:);%仅用三个波段
    img2=imread(f2);img2=img2(:,:,:);
    
    im_=single(cat(3,img2,img1));
    im_=(im_-vmin)./(vmax-vmin)+vmin;
    
    img3=imread(f3); 
    %制作标签 change 1 nochange 2
    tmp=img3; tmp(img3==255)=1; tmp(img3~=255)=2;

    label=single(tmp);
    % 测试
 net2.eval({'input',im_}) ;
scores = net2.vars(net2.getVarIndex('prediction')).value ;
scores = squeeze(gather(scores)) ;

[bestScore, best] = max(scores,[],3) ;
truth = uint8( label );
pre = uint8( best ); 
% end
% 计算准确率
accurcy = length(find(pre(:)==truth(:)))/length(truth(:));
disp(['accurcy = ',num2str(accurcy*100),'%']);

 subplot(1,2,1);imshow(truth,[]); subplot(1,2,2);;imshow(pre,[])
%  imwrite(pre,[n1,'.tif']);