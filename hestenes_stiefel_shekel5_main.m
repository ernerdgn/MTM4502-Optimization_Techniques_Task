function hestenes_stiefel_shekel5_main()
    a = [4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7; 
         4, 1, 8, 6, 3; 
         4, 1, 8, 6, 7];
    c = [0.1; 0.2; 0.2; 0.4; 0.4];

    lower_bound = 3.7;
    upper_bound = 4.3;

    x0 = lower_bound + (upper_bound - lower_bound) * rand(4, 1);

    tol = 1e-4;
    max_iter = 30;

    x = x0;
    fprintf("initial: [%f, %f, %f, %f]\n", x(1), x(2), x(3), x(4))
    tic;
    for iter = 1:max_iter
        [f_val, grad] = shekel5_and_gradient(x, a, c);
        
        if iter == 1
            d = -grad;
        else
            beta_hs = (grad' * (grad - grad_old)) / (d' * (grad - grad_old));
            d = -grad + beta_hs * d;
        end

        alpha = line_search(x, d, a, c);

        x = x + alpha * d;

        x = max(0, min(10, x));

        fprintf('iteration: %d\n', iter);
        fprintf('fval: %f\n', f_val);

        if norm(grad) < tol
            fprintf('Converged in %d iterations.\n', iter);
            break;
        end

        grad_old = grad;
    end
    
    if iter == max_iter
        fprintf('Maximum iterations reached.\n');
    end
    elapsedTime = toc;
    
    fprintf('iteration = %d\n', iter);
    fprintf('Minimum found at: [%f, %f, %f, %f]\n', x(1), x(2), x(3), x(4));
    disp(['The code took ', num2str(elapsedTime), ' seconds to run.'])
end

function [f_val, grad] = shekel5_and_gradient(x, a, c)
    f_val = 0;
    grad = zeros(4, 1);
    
    for i = 1:5
        diff = x - a(:, i);
        denom = sum(diff.^2) + c(i);
        
        f_val = f_val - 1 / denom;
        
        grad = grad + (2 * diff) / denom^2;
    end
end

function alpha = line_search(x, d, a, c)
    alpha = 1;
    rho = 0.5;
    c1 = 1e-4;
    
    [f_val, ~] = shekel5_and_gradient(x, a, c);
    
    while true
        x_new = x + alpha * d;
        x_new = max(0, min(10, x_new));
        
        [f_val_new, ~] = shekel5_and_gradient(x_new, a, c);
        
        if f_val_new <= f_val + c1 * alpha * d' * f_val
            break;
        end
        
        alpha = alpha * rho;
    end
end