function p = predictImg(Theta1, Theta2, filename)
colormap(gray);
img = imread(filename);
img = double(img);
imgVec = reshape(img',1,numel(img));
imgVec = (imgVec - mean(imgVec))/255;
imagesc(reshape(imgVec,[28, 28])');
p = predict(Theta1, Theta2, imgVec);
end