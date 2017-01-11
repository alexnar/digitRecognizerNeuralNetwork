function p = predictImg(Theta1, Theta2, filename)
colormap(gray);
img = imread(filename);
img = imresize(img, [28 28]);
size(img)
img = double(img);
imgVec = reshape(img',1,numel(img));
imagesc(reshape(imgVec,[28, 28])');
size(imgVec)
p = predict(Theta1, Theta2, imgVec/255);
end