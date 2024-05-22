function plot_algorithm_path()
    a = [4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7; 
         4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7];
    c = [0.1; 0.2; 0.2; 0.4; 0.4];

    % lower_bound = 0;
    % upper_bound = 10;
    % x0 = lower_bound + (upper_bound - lower_bound) * rand(4, 1);
    % x0 = [3.954450; 4.041713; 3.947645; 4.043338];
    % x0 = [3.950592; 4.064731; 4.045430; 4.014870];
    x0 = [4.057161; 4.036067; 3.940584; 4.058438];

    %[x_nr, path_nr] = newton_raphson(x0,a,c);
    %[x_hs, path_hs] = hestenes_stiefel(x0, a, c);
    %[x_pr, path_pr] = polak_ribiere(x0, a, c);
    [x_fr, path_fr] = fletcher_reeves(x0,a,c);

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
    %plot(path_nr(1,:), path_nr(2,:), 'g-*', 'LineWidth', 1);
    %plot(path_hs(1,:), path_hs(2,:), 'b-*', 'LineWidth', 1);
    %plot(path_pr(1,:), path_pr(2,:), 'b-*', 'LineWidth', 1);
    plot(path_fr(1,:), path_fr(2,:), 'b-*', 'LineWidth', 1);
    xlabel('x1');
    ylabel('x2');
    title('Optimization Path of the Fletcher-Reeves');
    legend('Shekel 5 Function', 'Fletcher-Reeves', 'Location', 'best');
    hold off;
end

function [x, path] = newton_raphson(x0, a, c)
    tol = 10^(-4);
    max_iter = 100;

    x = x0;
    path = x;
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    tic;
    for iter = 1:max_iter
        [f_val, grad, hess] = shekel5_and_gradient(x, a, c);
        % f_val = -f_val;
        dx = -hess \ grad;
        x = x + dx;
        
        x = max(0, min(10, x));
        path = [path, x];
        
        fprintf("iter: %d\n", iter);
        fprintf("x*: [%f, %f, %f, %f]\n",x(1), x(2), x(3), x(4));
        fprintf("f(x) = %f\n", f_val);
        % Check convergence
        if norm(dx) < tol
            fprintf('Converged in %d iterations.\n', iter);
            break;
        end
        fprintf("===========\n");
    end
    
    if iter == max_iter
        fprintf('Maximum iterations reached.\n');
    end
    elapsedTime = toc;
    fprintf('Minimum found at: [%f, %f, %f, %f]\n', x(1), x(2), x(3), x(4));
    disp(['The code took ', num2str(elapsedTime), ' seconds to run.\n']);
end

function [x, path] = polak_ribiere(x0, a, c)
    tol = 1e-4;
    max_iter = 100;
    x = x0;
    path = x;
    [f_val, grad] = shekel5_and_gradient(x, a, c);
    d = -grad;
    %fprintf("x at the start: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4));
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    tic;
    for iter = 1:max_iter
        alpha = line_search(x, d, a, c);
        x = x + alpha * d;
        
        fprintf("iter: %d\n", iter);
        fprintf("x*: [%f, %f, %f, %f]\n",x(1), x(2), x(3), x(4));
        path = [path, x];
        [f_val_new, grad_new] = shekel5_and_gradient(x, a, c);
        beta_pr = (grad_new' * (grad_new - grad)) / (grad' * grad);
        d = -grad_new + beta_pr * d;
        grad = grad_new;
        fprintf("f(x) = %f\n", f_val_new);
        if norm(grad) < tol
            break;
        end
        fprintf("===========\n");
    end
    %fprintf("iter: %f\n", iter);
    elapsedTime = toc;
    fprintf('Minimum found at: [%f, %f, %f, %f]\n', x(1), x(2), x(3), x(4));
    disp(['The code took ', num2str(elapsedTime), ' seconds to run.']);
end

function [x, path] = hestenes_stiefel(x0, a, c)
    tol = 1e-4;
    max_iter = 100;
    x = x0;
    
    [f_val, grad] = shekel5_and_gradient(x, a, c);
    d = -grad;
    grad_old = grad;
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    path = x;
    tic;
    for iter = 1:max_iter
        alpha = line_search(x, d, a, c);
        x = x + alpha * d;
        x = max(0, min(10, x));
        path = [path, x];
        [f_val, grad_new] = shekel5_and_gradient(x, a, c);
        
        fprintf("iter: %d\n", iter);
        fprintf("x*: [%f, %f, %f, %f]\n",x(1), x(2), x(3), x(4));
        
        delta_grad = grad_new - grad_old;
        beta_hs = (delta_grad' * grad_new) / (d' * delta_grad);
        d = -grad_new + beta_hs * d;
        
        grad_old = grad_new;
        fprintf("f(x) = %f\n", f_val);
        if norm(grad_new) < tol
            break;
        end
        fprintf("===========\n");
    end
    elapsedTime = toc;
    fprintf('Minimum found at: [%f, %f, %f, %f]\n', x(1), x(2), x(3), x(4));
    disp(['The code took ', num2str(elapsedTime), ' seconds to run.']);
end

function [x, path] = fletcher_reeves(x0, a, c)
    tol = 1e-4;
    max_iter = 100;
    x = x0;
    path = x;
    [f_val, grad] = shekel5_and_gradient(x, a, c);
    d = -grad;
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    tic;
    for iter = 1:max_iter
        alpha = line_search(x, d, a, c);
        x = x + alpha * d;
        path = [path, x];
        [f_val, grad_new] = shekel5_and_gradient(x, a, c);
        beta_fr = (grad_new' * grad_new) / (grad' * grad);
        d = -grad_new + beta_fr * d;
        grad = grad_new;
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
