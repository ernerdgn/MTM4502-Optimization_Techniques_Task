function plot_shekel5()
    a = [4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7; 
         4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7];
    c = [0.1; 0.2; 0.2; 0.4; 0.4];

    x3 = 5;
    x4 = 5;

    [x1_grid, x2_grid] = meshgrid(0:0.1:10, 0:0.1:10);
    f_val_grid = zeros(size(x1_grid));

    for i = 1:size(x1_grid, 1)
        for j = 1:size(x1_grid, 2)
            x = [x1_grid(i, j); x2_grid(i, j); x3; x4];
            f_val_grid(i, j) = shekel5(x, a, c);
        end
    end

    figure;
    surf(x1_grid, x2_grid, f_val_grid);
    title('Shekel 5 Function');
    xlabel('x1');
    ylabel('x2');
    zlabel('f(x1, x2, x3=5, x4=5)');
    colorbar;
end

function f_val = shekel5(x, a, c)
    f_val = 0;
    for i = 1:5
        diff = x - a(:, i);
        denom = sum(diff.^2) + c(i);
        f_val = f_val - 1 / denom;
    end
end