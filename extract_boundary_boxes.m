model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;


opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
%opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 50;  % max number of boxes to detect


fileID = fopen('predicted_bounding_box.txt','w');
for i=0:99
    image_name = ['../test/images/' num2str(i) '.JPEG' ];
    I = imread(image_name);
    bbs=edgeBoxes(I,model,opts);
    %length(bbs)
    for index=1:50
        x = bbs(index,1);
        y = bbs(index,2);
        w = bbs(index,3);
        h = bbs(index,4);
        x_last = x+w;
        y_last = y+h;
        boxes = [num2str(i) '.JPEG,' num2str(y) ',' num2str(x) ',' num2str(y_last) ',' num2str(x_last) '\n'];
        fprintf(fileID, boxes);
    end
    if(mod(i,5)==0)
        display('a')
        figure;
        imshow(I);
        hold on;
        axis on;
        title(i)
        for index=1:15
            x = bbs(index,1);
            y = bbs(index,2);
            w = bbs(index,3);
            h = bbs(index,4);
            rectangle('Position',[x,y,w,h],...
            'EdgeColor', rand(1,3),...
            'LineWidth', 3,...
            'LineStyle','-')
        end
        %drawnow;
        %pause(1);
        %hold off;
    end
end

fclose(fileID);
