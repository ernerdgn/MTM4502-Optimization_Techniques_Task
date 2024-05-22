function newton_raphson_shekel5()
    a = [4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7; 
         4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7];
    c = [0.1; 0.2; 0.2; 0.4; 0.4];

    %x0 = [3.9997;3.9997;3.9997;3.9997];
    %x0 = [3.99;3.99;3.99;3.99];
    %x0 = [3.95;3.94;3.94;3.94];
    %x0 = [4.05;4.07;4.05;4.06];
    lower_bound = 0;
    upper_bound = 10;

    x0 = lower_bound + (upper_bound - lower_bound) * rand(4, 1);

    tol = 10^(-4);
    max_iter = 100;

    x = x0;
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    tic;
    for iter = 1:max_iter
        [f_val, grad, hess] = shekel5(x, a, c);

        dx = -hess \ grad;
        x = x + dx;

        x = max(0, min(10, x));

        fprintf('iteration: %d\n', iter);
        fprintf('fval: %f\n', f_val);

        if norm(dx) < tol
            fprintf('Converged in %d iterations.\n', iter);
            break;
        end
    end
    
    if iter == max_iter
        fprintf('Maximum iterations reached.\n');
    end
    elapsedTime = toc;
    
    fprintf('iteration = %f\n', iter);
    fprintf('Minimum found at: [%f, %f, %f, %f]\n', x(1), x(2), x(3), x(4));
    disp(['The code took ', num2str(elapsedTime), ' seconds to run.'])
end

function [f_val, grad, hess] = shekel5(x, a, c)
    f_val = 0;
    grad = zeros(4, 1);
    hess = zeros(4, 4);
    
    for i = 1:5
        diff = x - a(:, i);
        denom = sum(diff.^2) + c(i);
        
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
