%测试模型
%unbalance的模型
imdb_unbalance.mat;
%balance样本
net = Main_model();
Main_train(net);


%%%%%%%%%%%%%%%%%
%%%%% unet %%%%%%%%%
unet = Unet_init();
Main_train(unet,'exp_unet');
