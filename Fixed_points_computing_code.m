% % Initialize Parallel Computing Pool
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local', numWorkers);  % Replace numWorkers with the desired number of workers
end


% Define the number of subnetworks and neurons in each subnetwork
numSubnetworks = 4;
numNeuronsPerSubnetwork = 3;
vectorLength=3*3;
uniformDecay=0.1;

syms x1 w1 x2 w2 x3 w3 x4 C real;
x1 = sym('x1', [numNeuronsPerSubnetwork, 1]);
w1 = sym('w1', [numNeuronsPerSubnetwork*numNeuronsPerSubnetwork, 1]);

x2 = sym('x2', [numNeuronsPerSubnetwork, 1]);
w2 = sym('w2', [numNeuronsPerSubnetwork*numNeuronsPerSubnetwork, 1]);

x3 = sym('x3', [numNeuronsPerSubnetwork, 1]);
w3 = sym('w3', [numNeuronsPerSubnetwork*numNeuronsPerSubnetwork, 1]);

x4 = sym('x4', [numNeuronsPerSubnetwork, 1]);
w4 = sym('w4', [numNeuronsPerSubnetwork*numNeuronsPerSubnetwork, 1]);

W121=10*randn(1);
W123=10*randn(1);
W231=10*randn(1);
W233=10*randn(1);
W342=10*randn(1);
W343=10*randn(1);
W411=10*randn(1);
W413=10*randn(1);

% for 3 neurons set every 3th element to zero
indicesToDelete = [1, 5:4:vectorLength];  
% sub network 1
sig_x1 = 1 ./ (1 + exp(-x1));
xx1 = (sig_x1 * sig_x1.');
xxx1= xx1(:);
w1_reshape=reshape(w1, [numNeuronsPerSubnetwork, numNeuronsPerSubnetwork]).';
dx1 =-uniformDecay*x1 + (w1_reshape-diag(diag(w1_reshape))) * sig_x1 +[W121*(1 ./ (1 + exp(-x2(1))));0;W123*(1 ./ (1 + exp(-x2(3))))];
dw1 =-uniformDecay*w1 - C* eye(9) * xxx1;
ddw1=dw1;
ddw1(indicesToDelete) = [];  % delete self-loops equations
vw1=w1;
vw1(indicesToDelete) = [];   % delete self-loops weights

% sub network 2
sig_x2 = 1 ./ (1 + exp(-x2));
xx2 = (sig_x2 * sig_x2.');
xxx2= xx2(:);
w2_reshape=reshape(w2, [numNeuronsPerSubnetwork, numNeuronsPerSubnetwork]).';
dx2 =-uniformDecay*x2 + (w2_reshape-diag(diag(w2_reshape))) * sig_x2 +[W231*(1 ./ (1 + exp(-x3(1))));0;W233*(1 ./ (1 + exp(-x3(3))))];
dw2 =-uniformDecay*w2 - C* eye(9) * xxx2;
ddw2=dw2;
ddw2(indicesToDelete) = [];
vw2=w2;
vw2(indicesToDelete) = [];

% sub network 3
sig_x3 = 1 ./ (1 + exp(-x3));
xx3 = (sig_x3 * sig_x3.');
xxx3= xx3(:);
w3_reshape=reshape(w3, [numNeuronsPerSubnetwork, numNeuronsPerSubnetwork]).';
dx3 =-uniformDecay*x3 + (w3_reshape-diag(diag(w3_reshape))) * sig_x3 +[0;W342*(1 ./ (1 + exp(-x4(2))));W343*(1 ./ (1 + exp(-x4(3))))];
dw3 =-uniformDecay*w3 - C* eye(9) * xxx3;
ddw3=dw3;
ddw3(indicesToDelete) = [];
vw3=w3;
vw3(indicesToDelete) = [];

% sub network 4
sig_x4 = 1 ./ (1 + exp(-x4));
xx4 = (sig_x4 * sig_x4.');
xxx4= xx4(:);
w4_reshape=reshape(w4, [numNeuronsPerSubnetwork, numNeuronsPerSubnetwork]).';
dx4 =-uniformDecay*x4 + (w4_reshape-diag(diag(w4_reshape))) * sig_x4 +[W411*(1 ./ (1 + exp(-x1(1))));0;W413*(1 ./ (1 + exp(-x1(3))))];
dw4 =-uniformDecay*w4 - C* eye(9) * xxx4;
ddw4=dw4;
ddw4(indicesToDelete) = [];
vw4=w4;
vw4(indicesToDelete) = [];

% redifined vector field and varaibles
vfield=[dx1;ddw1;dx2;ddw2;dx3;ddw3;dx4;ddw4];
vars = [x1;vw1;x2;vw2;x3;vw3;x4;vw4;];



%% sloving equilibria as parameters change using parallel computing
pts_num= 100;

% parameter range
hmin=-10;
hmax=10;
Mu = linspace(hmin,hmax,pts_num);


number_subsimul=40;

% eqlist=cell(length(Mu),number_subsimul,1);

% Initialize a cell array to store intermediate results
intermediateResults = cell(length(Mu), 1);


% Create a shared counter to track progress
progress = parallel.pool.DataQueue;
afterEach(progress, @(var) fprintf('Progress: %d/%d (%.2f%%)\n', var, length(Mu), (var / length(Mu)) * 100));


 parfor k=1:1:length(Mu)
    %fixed points computed for each mue 
    vfieldt = subs(vfield,C,Mu(k));
    tempEqlist = cell(number_subsimul, 1);
    for i=1:number_subsimul                  % this part is to do multiple searching
                                             % of eq in one trial of mue
        solutions = vpasolve(vfieldt, vars, 'random', true);   %calculating fixed points using vpasolve
        fieldNames = fieldnames(solutions);
        fieldValues = cell(1, numel(fieldNames));

    
        for j = 1:numel(fieldNames)
           fieldValues{j} = double(solutions.(fieldNames{j}));
        end
    

        tempEqlist{i} = fieldValues;
 
    end
        % Store the results in the intermediate cell array
    intermediateResults{k} = tempEqlist;
        % Update the shared counter to track progress
    send(progress, k);
    
 end
 
% Combine the intermediate results to form the final eqlist
eqlist = cell(length(Mu), 1);
for k = 1:length(Mu)
    eqlist{k} = cat(1, intermediateResults{k}{:});
end

% close the parallel computing pool when done
delete(poolobj);

% save the file
save('eqlist_rnnornn.mat', 'eqlist');


%% plot bifurcation diagram
% get equilibria list from saved file
eqlist = load("eqlist_rnnornn.mat");
fixedPoints = eqlist.eqlist;
numIterations = 120;
numSearch = 40;
numVariables =36;

% Initialize a cell array to store the fixed points as matrices
fixedPointsMatrix = cell(numIterations, 1);
for i = 1:numIterations
    for j = 1:1
        % Convert each cell to a matrix and store it in the cell array
        fixedPointsMatrix{i, j} = cell2mat(fixedPoints{i, j});
    end
end

pts_num= numIterations;
Cmin=-8;
Cmax=2;
Mu = linspace(Cmin,Cmax,pts_num);

% plot
f=figure('Visible','on'); 
hold on;
axis tight;
for k=1:1:numIterations
    for n=1:1:length(fixedPointsMatrix{k, 1}(:,1))
       x11eq=fixedPointsMatrix{k, 1}(n,4);
       x21eq=fixedPointsMatrix{k, 1}(n,30);
       % plot the equilibrium points when the values are not empty, otherwise display them
       if ~isempty(x11eq) && ~isempty(x21eq)
            plot3(Mu(k),x11eq,x21eq,'.b','MarkerSize',10);
       else 
            disp(['xeq ', num2str(k), ' ', num2str(n) ' is empty.']);
       end
    end
end
xlim([-8,2]);
zlim([-60,10]);
% parameter axis
xlabel('C','FontSize',15); 
% two arbitrary states axes
ylabel('x_{11}','FontSize',15); 
zlabel('x_{21}','FontSize',15);
set(gca,'FontSize',15);
grid on;
view([45 25]) ;
