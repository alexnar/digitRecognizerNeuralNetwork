function scaled_data = featureScaling(data)

% scaled_data = (data - mean(data(:)))/(max(data(:)) - min(data(:)));

scaled_data = data / 255; % It is enough scaling for greyscale image

end
