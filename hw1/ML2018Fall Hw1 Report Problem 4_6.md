# ML2018Fall Hw1 Report Problem 4~6

## 4 (1%)
### (4-a)
Given $t_n$ is the data point of the data set $\mathcal D=\{t_1, ...,t_N \}$ . Each data point $t_n$ is associated with a weighting factor $r_n$ > 0.
The sum-of-squares error function becomes:
$$E_D(\mathbf w) = \frac 1 2 \sum_{n=1}^{N}r_n(t_n -\mathbf w^{\mathbf T}\mathbf x_n)^2$$
Find the solution $\mathbf w^{*}$ that minimizes the error function.

### (4-b)
Following the previous problem(2-a), if
$$\mathbf t = [t_1 t_2 t_3] = \begin{bmatrix}
       0  &10&5\\ 
     \end{bmatrix},
     \mathbf X=[\mathbf {x_1 x_2 x_3}] = \begin{bmatrix}
       2 & 5 &5          \\[0.3em]
       3 & 1 & 6         \\[0.3em]
     \end{bmatrix}$$
     $$r_1 = 2, r_2 = 1, r_3 = 3$$
Find the solution $\mathbf w^{*}$ .

## 5 (1%)
Given a linear model:

$$ y(x, \mathbf w) = w_0 + \sum_{i=1}^{D}w_ix_i$$

with a sum-of-squares error function:

$$E(\mathbf w) = \frac 1 2 \sum_{n=1}^{N} \big(y(x_n, \mathbf w) -t_n ) \big)^2 $$

where $t_n$ is the data point of the data set $\mathcal D=\{t_1, ...,t_N \}$

Suppose that Gaussian noise $\epsilon_i$ with zero mean and variance $\sigma^2$ is added independently to each of the input variables $x_i$.
By making use of $\mathbb E[\epsilon_i \epsilon_j] = \delta_{ij} \sigma^2$ and $\mathbb E[\epsilon_i] = 0$, show that minimizing $E$ averaged over the noise distribution is equivalent to minimizing the sum-of-squares error for noise-free input variables with the addition of a weight -decay regularization term, in which the bias parameter $w_0$ is omitted from the regularizer.

Hint
*   $$\delta_{ij} = \begin{equation}
  \left\{
   \begin{array}{c}
   1 (i=j),  \\
   0 (i\neq j).  \\
   \end{array}
  \right.
  \end{equation}
$$

## 6 (1%)
$\mathbf A \in \mathbb R^{n \times n},  \alpha$ is one of the elements of $\mathbf A,$ prove that

$$\frac {\mathrm d} {\mathrm d \alpha } ln|\mathbf A| = Tr\bigg(\mathbf A^{-1} \frac {\mathrm d} {\mathrm d \alpha} \mathbf A \bigg) $$
where the matrix $\mathbf A$ is a real, symmetric, non-sigular matrix.

Hint:
*   The determinant and trace of $\mathbf A$ could be expressed in terms of its eigenvalues.



## 7(不算分，自行嘗試)
在第6中，若 $\mathbf A$不為symmetric，亦可推導出類似形式關係，可嘗試證明general case的推導，此部分不算分。