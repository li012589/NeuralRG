## Simplectic

$$
\Q^{T}J\Q = J \\
\begin{align}
\Q &= exp(Q)\\
&= (I+\frac{Q}{N})^{N}
\end{align}
\\
q = \frac{Q}{N} \\
J = 
\begin{bmatrix}
0&I \\
-I&0\\
\end{bmatrix}
\\
q = 
\begin{bmatrix}
A & B \\
C & D \\
\end{bmatrix}
\\ 
\Q^TJ\Q = J\\
\Rightarrow((I+q)^T)^NJ(I+1)^N = J \\
\Rightarrow(I+q)^TJ(I+q)=J \\
\Rightarrow
q^TJ+Jq+O(q^2)=0\\
\Rightarrow
\begin{cases}
C = C^T \\
A^T = -D \\
B = B^T
\end{cases}\\
\Rightarrow q = 
\begin{bmatrix}
A&B\\
C&-A^T\\
\end{bmatrix}\\
where\ C = C^T; B = B^T
$$



## Orthogonal

$$
\Q^{T}\Q = I \\
\begin{align}
\Q &= exp(Q)\\
&= (I+\frac{Q}{N})^{N}
\end{align}
\\
q = \frac{Q}{N} \\
I = 
\begin{bmatrix}
I&0 \\
0&I\\
\end{bmatrix}
\\
q = 
\begin{bmatrix}
A & B \\
C & D \\
\end{bmatrix}
\\ 
\Q^T\Q = I\\
\Rightarrow((I+q)^T)^N(I+1)^N = I \\
\Rightarrow(I+q)^T(I+q)=I \\
\Rightarrow
q^T+q+O(q^2)=0\\
\Rightarrow
\begin{cases}
D^T = -D \\
A^T = -A \\
B = -C^T
\end{cases}\\
\Rightarrow q = 
\begin{bmatrix}
A&B\\
-B^T&D\\
\end{bmatrix}\\
where\ A = -A^T; D = -D^T
$$

for 2$\times$2 :
$$
q = 
\begin{bmatrix}
0&b \\
-b&b
\end{bmatrix}\\
\Rightarrow
\Q = 
\begin{bmatrix}
\cos\theta &\sin\theta\\
-\sin\theta &\cos\theta
\end{bmatrix}
$$
