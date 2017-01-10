function displayDigit = displayDigit(X, y, index)
colormap(gray);
image_height = 28;
image_length = 28;
imagesc(reshape(X(index,:),[image_height, image_length])');
title(['\fontsize{16}It is: ', num2str(y(index))])
end