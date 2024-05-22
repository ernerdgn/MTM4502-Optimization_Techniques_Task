function plot_algorithm_path_gradient_descent()
    a = [4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7; 
         4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7];
    c = [0.1; 0.2; 0.2; 0.4; 0.4];

    lower_bound = 0;
    upper_bound = 10;

    x0 = lower_bound + (upper_bound - lower_bound) * rand(4, 1);
    x0 = [3.950592; 4.064731; 4.045430; 4.014870];

    [x_gd, path_gd] = steepest_descent(x0,a,c);

    [X, Y] = meshgrid(0:0.1:10, 0:0.1:10);
    Z = zeros(size(X));
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            Z(i, j) = -shekel5([X(i, j); Y(i, j); 5; 5], a, c);
        end
    end
    
    figure;
    contour(X, Y, Z, 30);
    hold on;
    plot(path_gd(1,:), path_gd(2,:), 'g-*', 'LineWidth', 1);
    xlabel('x1');
    ylabel('x2');
    title('Optimization Path of the Gradient Descent');
    legend('Shekel 5 Function', 'Steepest Descent', 'Location', 'best');
    hold off;
end

function [x, path] = steepest_descent(x0, a, c)
    tol = 1e-4;
    max_iter = 100;
    x = x0;
    path = x;
    [f_val, grad] = shekel5_and_gradient(x, a, c);
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    tic;
    for iter = 1:max_iter
        d = -grad;
        alpha = line_search(x, d, a, c);
        x = x + alpha * d;
        path = [path, x];
        [f_val, grad] = shekel5_and_gradient(x, a, c);
        fprintf("iter: %d\n", iter);
        fprintf("x*: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4));
        fprintf("f(x) = %f\n", f_val);
        if norm(grad) < tol
            break;
        end
        fprintf("===========\n");
    end
    elapsedTime = toc;
    fprintf('Minimum found at: [%f, %f, %f, %f]\n', x(1), x(2), x(3), x(4));
    disp(['The code took ', num2str(elapsedTime), ' seconds to run.']);
end

function [f_val, grad, hess] = shekel5_and_gradient(x, a, c)
    f_val = 0;
    grad = zeros(4, 1);
    hess = zeros(4, 4);
    
    for i = 1:5
        diff = x - a(:, i);
        denom = sum(diff.^2) + c(i);
        
        if denom == 0
            denom = eps;
        end
        
        f_val = f_val - 1 / denom;

        grad = grad + (2 * diff) / denom^2;

        for j = 1:4
            for k = 1:4
                if j == k
                    hess(j, k) = hess(j, k) + (2 / denom^2) - (8 * diff(j)^2 / denom^3);
                else
                    hess(j, k) = hess(j, k) - (8 * diff(j) * diff(k) / denom^3);
                end
            end
        end
    end
end

function alpha = line_search(x, d, a, c)
    alpha = 1;
    rho = 0.5;
    c1 = 1e-4;
    
    [f_val, grad] = shekel5_and_gradient(x, a, c);
    
    while true
        x_new = x + alpha * d;
        x_new = max(0, min(10, x_new));
        
        [f_val_new, ~] = shekel5_and_gradient(x_new, a, c);
        
        if f_val_new <= f_val + c1 * alpha * (grad' * d)
            break;
        end
        
        alpha = alpha * rho;
    end
end

function f_val = shekel5(x, a, c)
    f_val = 0;
    for i = 1:5
        diff = x - a(:, i);
        denom = sum(diff.^2) + c(i);
        f_val = f_val - 1 / denom;
    end
end
