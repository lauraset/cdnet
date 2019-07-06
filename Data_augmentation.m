function  Data_augmentation(im,varargin )
%DATA_AUGMENTATION Summary of this function goes here
%   Detailed explanation goes here

opts=[];
opts.savepath=[];%save path for augmented images
opts.Flip = false ;% flip 2:up-down and left-right 
opts.Aspect = 30 ;%rotation 11: 30,60,90,120,150,180,210,240,270,300,330
opts.Scale = 1 ;% scaling 2: 4m 6m
opts.Color = 0 ; %color 1: HSV
opts.Noise = 0; %noise 1; gussain
opts.Erase = 0; %data erase: 1 delete data [9,9] 大概对应40m x 40m相当于丢失了上下文 数据的可用性降低

opts = vl_argparse(opts, varargin);

% data augmentation
%noise
vmax=max(im(:));
vmin=min(im(:));
im_=(im-vmin)./(vmax-vmin);
im_=imnoise(im,'gaussian',0,0.01); %default value
im_=im_*(vmax-vmin)+vmin;

% rotate
num=floor(360/opts.Aspect)-1;
for i=1:(360/opts.Aspect-1)
    im_=imrotate(im,i*opts.Aspect);
    imwrite([name,path],im_);
end

%resize
im_=imresize(im,0.5);im_=imresize(im_,2);

im_=imresize(im,0.25);im_=imresize(im_,4);

%flip
im_=flipup(im);
im_=fliplr(im);

%erase randomly
sd=4;seed=false;
while(~seed)
xp=floor(rand(1)*size(im,1));
yp=floor(rand(1)*size(im,2));
if xp-sd>0 &&  xp+sd<size(im,1) && yp-sd>0 &&  yp+sd<size(im,2) 
    seed=true;
end
end
im_=im;im_(xp-sd:xp+sd,yp-sd:yp+sd)=0;%erase

%color

end

% Define the value for color shifting
shifting_color = [ 20, -20,  5, 50;
                  -20,  20,  0,  0;
                   20,  20, 50, -10;
                   20,  20, 50, -10];
                          
    % Shift the color channel
    im_ = color_shifting(im, shifting_color(1,i), shifting_color(2,i), shifting_color(3,i));

function [img_shiftcolor]=color_shifting(img, shift_red, shift_green, shift_blue)
    img_shiftcolor = img;
    img_shiftcolor(:,:,1)=img_shiftcolor(:,:,1)+shift_blue;
    img_shiftcolor(:,:,2)=img_shiftcolor(:,:,2)+shift_green;
    img_shiftcolor(:,:,3)=img_shiftcolor(:,:,3)+shift_red;
    img_shiftcolor(:,:,4)=img_shiftcolor(:,:,4)+shift_nir;

end

function [center_img] = center_crop(img, width, height, shiftwidth, shiftheight )
    img_size = size(img);
    heightbot = fix( (img_size(1)-height)/2 + shiftheight);
    heighttop = heightbot+height;
    widthleft = fix((img_size(2)-width)/2 + shiftwidth);
    widthright = widthleft+width;
    center_img = img(heightbot:heighttop, widthleft:widthright,:);
end

function [r, g, b] = get_color_channel(img)
    r=img(:,:,1);
    g=img(:,:,2);
    b=img(:,:,3);
    temp = zeros(size(img, 1), size(img, 2));
    r = cat(3, r, temp, temp);
    g = cat(3, temp, g, temp);
    b = cat(3, temp, temp, b);
end

