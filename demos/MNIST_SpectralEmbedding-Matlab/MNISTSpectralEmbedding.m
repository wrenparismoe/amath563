%% MNIST spectral embedding 

clc 
clear all
close all


%% read and visualize different MNIST graphs

MNISTimgfile = './train-images-idx3-ubyte';
MNISTlabelfile = './train-labels-idx1-ubyte';


% import images and labels
[images labels] = readMNIST(MNISTimgfile, MNISTlabelfile, 1000, 0);


% extract index of pairs of digits 
ind4 = find(labels == 3); 
ind8 = find(labels == 1); 

ind = [ind4; ind8];


for i=1:length(ind)
    
   x(:, i) = reshape(images(:, :, ind(i)), [20*20,1]); 
    
end


% compute pairwise distances between the x's

xdis = squareform( sqrt( (pdist(x').^2)./200 ) );

%% Proximity graphs

kappa_eps = @(t, epsilon)  (t <= epsilon);
kappa_Gaussian = @(t, epsilon) exp(- (t.^2)./(2*epsilon.^2) );

% compute weight matrices
epps = .7; 
W_eps = kappa_eps( xdis, epps ); 
W_eps = W_eps - diag(diag(W_eps));
W_Gaussian = kappa_Gaussian( xdis, epps); 
W_Gaussian = W_Gaussian - diag(diag(W_Gaussian));

% plot graphs 
g_eps = graph(W_eps);
LWidths_eps = g_eps.Edges.Weight/max(g_eps.Edges.Weight);

[U, S, V] = svd(x);
pca_coor= U'*x;

figure(1)
gplot( W_eps, [pca_coor(1,:); pca_coor(2,:)]');

hold on 

scatter( pca_coor(1,1:length(ind4)),  pca_coor(2,1:length(ind4)), 'r', 'filled')
scatter( pca_coor(1,length(ind4):end),  pca_coor(2,length(ind4):end), 'k', 'filled')

xlabel('Mode 1')
ylabel('Mode 2')

title('Digit 3 vs  digit 1 - epsilon graph')

hold off

% figure(2)
% gplot( W_Gaussian, [pca_coor(1,:); pca_coor(2,:)]');
% 

%% k-NN graph
KK = 10;
indknn = knnsearch( x', x', 'K', KK + 1 );

W_KNN = zeros( size(W_eps));


for i =1:length(ind)
   
    for j =1:KK
        
        W_KNN(i, indknn(i, j+1)) = kappa_Gaussian( norm( x(:,i) - x(:, indknn(i, j+1)) )./sqrt(200), epps );
        W_KNN(indknn(i, j+1), i) = W_KNN(i, indknn(i, j+1));
        
    end
    
end



figure(2)
gplot( W_KNN, [pca_coor(1,:); pca_coor(2,:)]');

hold on 

scatter( pca_coor(1,1:length(ind4)),  pca_coor(2,1:length(ind4)), 'r', 'filled')
scatter( pca_coor(1,length(ind4):end),  pca_coor(2,length(ind4):end), 'k', 'filled')

xlabel('Mode 1')
ylabel('Mode 2')

title('Digit 3 vs  digit 1 - KNN graph K = 5')

hold off



% Contruct unnormalized graph Laplacians on each graph 

D_eps = diag(sum(W_eps,2));
L_eps =  D_eps - W_eps; 


D_Gaussian = diag(sum(W_Gaussian,2 )); 
L_Gaussian = D_Gaussian - W_Gaussian;

D_KNN = diag(sum(W_KNN,2));
L_KNN = D_KNN - W_KNN; 


% compute eigenpairs

[Vphi_eps, Lambda_eps] = eig(L_eps); 
[Vphi_Gaussian, Lambda_Gaussian] = eig(L_Gaussian); 
[Vphi_KNN, Lambda_KNN] = eig(L_KNN); 

[diag(Lambda_eps), diag(Lambda_Gaussian), diag(Lambda_KNN)]


figure(3) 
plot( [diag(Lambda_eps), diag(Lambda_Gaussian), diag(Lambda_KNN)], 'LineWidth', 4 )
legend('Epsilon graph', 'Connected graph', 'KNN graph')
xlabel('Index j')
ylabel('Eigenvalue \lambda_j')
title('Unnormalized laplacian')

% construct normalized Laplacians 

L_eps_norm = (D_eps^(-1/2))*L_eps*(D_eps^(-1/2)); 
L_Gaussian_norm = (D_Gaussian^(-1/2))*L_Gaussian*(D_Gaussian^(-1/2));
L_KNN_norm = (D_KNN^(-1/2))*L_KNN*(D_KNN^(-1/2));


% compute normalized eigenpairs 


[Vphi_eps_norm, Lambda_eps_norm] = eig(L_eps_norm); 
[Vphi_Gaussian_norm, Lambda_Gaussian_norm] = eig(L_Gaussian_norm); 
[Vphi_KNN_norm, Lambda_KNN_norm] = eig(L_KNN_norm); 

[diag(Lambda_eps_norm), diag(Lambda_Gaussian_norm), diag(Lambda_KNN_norm)]


figure(3) 
plot( [diag(Lambda_eps_norm), diag(Lambda_Gaussian_norm), diag(Lambda_KNN_norm)], 'LineWidth', 4 )
legend('Epsilon graph', 'Connected graph', 'KNN graph')
xlabel('Index j')
ylabel('Eigenvalue \lambda_j')
title('Unnormalized laplacian')




%% embed the data using the Laplacian embedding for KNN graph 

figure(4)
set(gca,'FontSize',20)
scatter3( Vphi_KNN(1:length(ind4), 1),  Vphi_KNN(1:length(ind4), 2), Vphi_KNN(1:length(ind4), 3), 'r', 'filled')
hold on
scatter3( Vphi_KNN(length(ind4):end, 1),  Vphi_KNN(length(ind4):end, 2), Vphi_KNN(length(ind4):end, 3), 'k', 'filled') 
xlabel('varphi_1(j)')
ylabel('varphi_2(j)')
zlabel('varphi_3(j)')
legend('Digit 3', 'Digit 1');
title('Unnormalized Laplacian on KNN graph')
hold off

figure(5)
set(gca,'FontSize',20)
scatter3( Vphi_KNN_norm(1:length(ind4), 1),  Vphi_KNN_norm(1:length(ind4), 2), Vphi_KNN_norm(1:length(ind4), 3), 'r', 'filled')
hold on
scatter3( Vphi_KNN_norm(length(ind4):end, 1),  Vphi_KNN_norm(length(ind4):end, 2), Vphi_KNN_norm(length(ind4):end, 3), 'k', 'filled') 
xlabel('varphi_1(j)')
ylabel('varphi_2(j)')
zlabel('varphi_3(j)')
legend('Digit 3', 'Digit 1');
title('Normalized Laplacian on KNN graph')
hold off

