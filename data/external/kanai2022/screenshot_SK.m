function screenshot_SK(folderpath)

if(exist('folderpapth')) == 0
    folderpath = 'E:\0temp\MATLAB\ss\';
end
% Take screen capture
robot = java.awt.Robot();
pos = [1, 1, 3840, 2160]; % [left top width height]
rect = java.awt.Rectangle(pos(1),pos(2),pos(3),pos(4));
cap = robot.createScreenCapture(rect);
% Convert to an RGB image
rgb = typecast(cap.getRGB(0,0,cap.getWidth,cap.getHeight,[],0,cap.getWidth),'uint8');
imgData = zeros(cap.getHeight,cap.getWidth,3,'uint8');
imgData(:,:,1) = reshape(rgb(3:4:end),cap.getWidth,[])';
imgData(:,:,2) = reshape(rgb(2:4:end),cap.getWidth,[])';
imgData(:,:,3) = reshape(rgb(1:4:end),cap.getWidth,[])';
% Show or save to file
% imshow(imgData)
imwrite(imgData,[folderpath 'screenshot_' char(datetime('now','Format','yyyyMMdd''T''HHmmss')) '.png']);
end