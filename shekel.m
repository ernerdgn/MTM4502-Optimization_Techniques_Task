function [y] = shekel(xx)

m = 5;
b = 0.1 * [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]';
C = [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
     4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6;
     4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0;
     4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6];

outer = 0;
for ii = 1:m
	bi = b(ii);
	inner = 0;
	for jj = 1:4
		xj = xx(jj);
		Cji = C(jj, ii);
		inner = inner + (xj-Cji)^2;
	end
	outer = outer + 1/(inner+bi);
end

y = -outer;

end