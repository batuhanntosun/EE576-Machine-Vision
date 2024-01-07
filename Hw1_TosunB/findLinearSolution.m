fileID = fopen("img1.txt");
formatSpec = '%d %d';
img1_coords = fscanf(fileID, formatSpec, [2 Inf]);
img1_coords = img1_coords';
[N,M] = size(img1_coords);
img1_coords = [img1_coords ones(N,1)];
fclose(fileID);
fileID2 = fopen("img2.txt");
img2_coords = fscanf(fileID2, formatSpec,[2 Inf]);
img2_coords = img2_coords';
fclose(fileID2);

% construct the matrix A and a

A = zeros(2*N,6);
a = zeros(2*N,1);
for i=1:N
    A(2*i-1,1:3) = img1_coords(i,:); 
    A(2*i,4:6) = img1_coords(i,:);
    
    a(2*i-1:2*i,1) = img2_coords(i,:)';
end

% solve Ah = a

first = (A'*a);
second = (A'*A);
h = second\first;

% create H matrix
H = zeros(3,3);
H(1,:)=h(1:3);
H(2,:)=h(4:6);
H(3,3)=1;
display(H);


