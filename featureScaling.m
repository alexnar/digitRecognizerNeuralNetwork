function scaled_data = featureScaling(data)

scaled_data = (data - mean(data(:)))/(max(data(:)) - min(data(:)));


end
