%����ģ��
%unbalance��ģ��
imdb_unbalance.mat;
%balance����
net = Main_model();
Main_train(net);


%%%%%%%%%%%%%%%%%
%%%%% unet %%%%%%%%%
unet = Unet_init();
Main_train(unet,'exp_unet');
