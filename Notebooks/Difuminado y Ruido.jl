### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 402b40b3-ad31-4561-8880-adb973772f32
begin
	using Pkg
	Pkg.add(["Colors","ColorVectorSpace","ImageShow","FileIO","ImageIO","HypertextLiteral","Images","Statistics","Distributions","LinearAlgebra","StatsBase","StatsPlots"])
	using Colors,ColorVectorSpace,ImageShow,FileIO,ImageIO
	using HypertextLiteral
	using Images, ImageShow 
	using Statistics,  Distributions, LinearAlgebra
	using StatsBase
end

# ╔═╡ 1e758897-b3f6-4813-812c-70f2ee88a3b8
begin
	Pkg.add(["Optimization","Zygote"])
	using Optimization, Zygote
	function total_var_segment(A,lambda)
		B = Float64.(channelview(A))
	    M, N = size(B)
		tam=M*N
		a=vec(B)
		f(x,_)=sum((a.-x).^2)+lambda*sum((x[i]-x[i-M])^2 for i in M+1:tam) + lambda*sum((x[i]-x[i+1])^2 for i in 2:tam if i%M !=0)
		x0=a
		p=0
		optf = OptimizationFunction(f, AutoZygote())
		prob = OptimizationProblem(optf, x0,p)
		sol=solve(prob, Optimization.LBFGS())
		B=reshape(sol.minimizer, size(A))
		return Gray.(B)
	end

	function total_variation(A,lambda)
		A = Float64.(channelview(A))
	    submatrix_size = 20
	    n, m = size(A)
	    B = copy(A)
	    for i in 1:submatrix_size:n
        	for j in 1:submatrix_size:m
            	fi = min(i + submatrix_size - 1, n)
            	fj = min(j + submatrix_size - 1, m)
            	submatrix = A[i:fi, j:fj]
            	B[i:fi, j:fj] = total_var_segment(submatrix, lambda)
        	end
    	end
	    return Gray.(B)
	end
end

# ╔═╡ d9d4169d-0af8-43e5-9611-7cf8ff69e5cd
md"""
Cuaderno realizado por Jhon Acosta, Yessica Trujillo, Carlos Nosa y Juan Diego Murcia 
"""

# ╔═╡ e0fcabe0-b120-11ef-1869-cf0f1342a959
md"""Vamos a usar las siguientes librerías:"""

# ╔═╡ 6c455d3d-f1e5-4c40-8724-a18d30166f36
md"""
# Introducción
"""

# ╔═╡ 899b6b24-4e39-491c-b4b2-b4484cb5e2d4
md"""
En este capítulo, se explorarán diversas transformaciones de restauración de imágenes y métodos de detección de bordes. Durante el proceso, se observará que, aunque estas transformaciones parezcan muy distintas entre si, todas están relacionadas matemáticamente a través de la operación de convolución.
"""

# ╔═╡ 8593931b-f918-4395-a600-ef51e151dd32
md"""
# Ruido en imágenes
"""

# ╔═╡ 3cda7462-9628-4006-b39d-f1cfbf61713e
md"""
A veces, una imágen (y, en general, cualquier tipo de señal), se ve afectada por una perturbación aleatoria, que resulta, por ejemplo, de interferencias o falta de memoria en los dispositivos. A este componente se le denomina *ruido*. A continuación, se ejemplifican distintos tipos.
"""

# ╔═╡ 4c606d57-c4a8-465b-a043-f89ee4f491b7
md"""##   $\cdot$ Ruido Blanco Gaussiano Aditivo (AWGN)"""

# ╔═╡ f4151cb3-8097-4da0-9f2d-a52118e23e37
md"""
En procesamiento de imagenes el **ruido** se puede definir como una variacion aleatoria no deseada en el brillo o el color de los pixeles. Este ruido generalmente se $\underline{\text{ modela }}$ agregando variables aleatorias que se distribuyen de manera independiente sobre los valores de los pixeles. 

Aunque el ruido puede provenir de diversas fuentes naturales, el modelo de ruido más común en procesamiento de señales e imágenes es el Ruido Blanco Gaussiano Aditivo **(AWGN)**. Este ruido se simula añadiendo variables aleatorias normales de **media cero** y una desviación estándar especificada a los valores de los píxeles.
"""

# ╔═╡ f2b3bbd9-2bd6-4d48-86db-93fbf736a20c
md""" Empleamos una conocida foto de Albert Einstein como ejemplo de imagen corrompida por **AWGN**. Usamos la funcion *randn* que genera numeros aleatorios con un distribucion normal y una desviacion estandar de $\sigma = 30$. 

"""

# ╔═╡ 41fa8758-47ef-400a-97dd-2e45fda34391
begin
	url="https://github.com/ytrujillol/Procesamiento-de-imagenes/blob/main/Images/Einstein.jpg?raw=true"
	fname = download(url)
	A = Gray.(load(fname));
end

# ╔═╡ 0d8d6355-460e-404d-b6ce-713c0c6ed713
md"""
La siguiente funcion corrompe imagenes con ruido Gaussiano. 
"""

# ╔═╡ 9afafb0f-ac0b-470b-aab3-b41e0a0dc0f3
function ruido_gaussiano(A, sigma) 
    M, N = size(A)               
    AWGN = sigma * randn(M, N) #sigma entre (0,1) pues la imagen esta normalizada 
    An = A + AWGN                 
    return An
end


# ╔═╡ 11076927-b844-4261-8069-6e183f60844c
[A ruido_gaussiano(A,0.2)]

# ╔═╡ 7f82bbae-a8ae-46a8-8fc3-92dd23079590
md"""##   $\cdot$ Ruido Sal y Pimienta"""

# ╔═╡ 54390a0b-7335-4437-8e48-ee3391ca15b4
md"""
Es un ruido que se caracteriza por la presencia de pixeles blancos y negros en lugares aislados de la imagen. Puede presentarse por variaciones agudas y repentinas en la señal de la imagen. Se puede presentar por errores en la transmisión, elementos de pixel dañados en una cámara, o falta de capacidad de memoria. Su nombre se debe a que la imagen parece haber sido "cubierta" por sal y pimienta (*Wikipedia*).
"""

# ╔═╡ de95337f-073d-4cab-ad17-ac8e90192de4
function sal_y_pimienta(A, prob) 
    M, N = size(A)  
    imagen_ruidosa = copy(A) 
    for i in 1:M
        for j in 1:N
            rand_val = rand()
            if rand_val < prob/2
                imagen_ruidosa[i, j] = 0  #Pixel blanco
            elseif rand_val > 1 - prob/2
                imagen_ruidosa[i, j] = 1  #Pixel negro
            end
        end
    end
    return imagen_ruidosa
end


# ╔═╡ 1beec73a-f89d-4c68-9642-887a064e983a
[A sal_y_pimienta(A,0.2)]

# ╔═╡ b8b84a35-4b7d-47be-ac66-0b2d490d5b65
md"""
## Ruido Speckle
"""

# ╔═╡ 3dfcbe46-a26a-484e-ad2d-f25dca23716b
md"""
Resulta de interferencia aleatoria, y, formalmente, no es ruido, sino un resultado de reflexiones difusas. Suele presentarse en radares, radares de apertura sintética, ultrasonido en medicina, y tomografías de coherencia óptica (*Wikipedia*). En el presente código, se genera al multiplicar el valor del pixel por una realización de $N(1,\sigma)$.
"""

# ╔═╡ d163328a-74cc-453c-9dbc-ee3b14ced35a
function agregar_ruido_speckle(A, sigma)
    M, N = size(A)  # Obtener las dimensiones de la imagen
    ruido = sigma * randn(M, N)  # Generar ruido normal multiplicativo
    noisy_image = A .* (1 .+ ruido)  # Multiplicar la imagen por el ruido
    
    return noisy_image
end


# ╔═╡ d02b4097-e9be-40f4-a52b-dc54ecd1fb2c
[A agregar_ruido_speckle(A,0.3)]

# ╔═╡ cb8d7d58-d0f8-43ed-a168-a292d7880f6d
md"""#  $\cdot$ Difuminado"""

# ╔═╡ 340808ee-0131-4c33-b17e-2b35067f93ae
md"""
Generalmente preferimos imagenes nitidas a las borrosas. Por ello desarrollamos tecnicas poderosas para agudizar imagenes. No obstante, en algunas ocasiones es necesario tambien hacer que una imagen parezca borrosa para algunas funciones como:

- Proteger la privacidad.
- Crear ilusión de mayor profundidad visual.
- Fotografía artística.
- Resaltar partes de imágenes.

Un método sencillo, y que igual funciona para el difuminado de imágenes, consiste en hallar un nuevo valor para cada pixel tomando el promedio de su valor con el de sus vecinos, por ejemplo, en una malla $3\times 3$, aplicamos sobre una imagen $A$ una transformación que reulta en una imagen $B$ (la versión borrosa de $A$) descrita como

$B\left[i,j\right]=\frac{1}{9}\sum_{k=-1}^{1}\sum_{l=-1}^{1}A\left[i+k,j+l\right]$

A continuación, se aplica difuminado sobre nuestra imagen de ejemplo.
"""

# ╔═╡ cc5d32c4-b124-4a5b-af7f-a69be2c309e1
function imagen_difuminada(A)
    A = Float64.(channelview(A))
    M, N = size(A)
    for m in 2:M-1
        for n in 2:N-1
            A[m, n] = (A[m-1, n-1] + A[m, n-1] + A[m+1, n-1] +
                       A[m-1, n] + A[m, n] + A[m+1, n] +
                       A[m-1, n+1] + A[m, n+1] + A[m+1, n+1]) / 9
        end
    end
	return Gray.(A)
end

# ╔═╡ 4943534b-e2a0-4733-b630-0f743df58f6f
[A imagen_difuminada(A)]

# ╔═╡ 417e745f-6380-472c-bfe2-18bb92f42a5f
md"""
Una de las principales aplicaciones del difuminado de imagenes es la eliminación de ruido. A continuación, intentaremos reducir cada uno de los tres ruidos generados anteriormente con este método "sencillo" e "ingenuo" de promedio sin ponderar.
"""

# ╔═╡ 9bfdce84-9554-41cf-a042-97d730c5471c
A1 = ruido_gaussiano(A,0.2);

# ╔═╡ d34c0c51-e87a-4732-bea5-ceb01a74aefa
A2 = sal_y_pimienta(A,0.2);

# ╔═╡ e9a3d6ae-52b0-4949-8985-5cb3f890474a
A3 = agregar_ruido_speckle(A, 0.3);

# ╔═╡ 226c10a9-2cd1-4683-8aff-38db6d54dc72
[A1 imagen_difuminada(A1)]

# ╔═╡ 3aa265d0-44d7-4e4b-aae2-b7b9d628d04b
[A2 imagen_difuminada(A2)]

# ╔═╡ aae5b170-5979-4b32-b92a-f0e5e563aa29
[A3 imagen_difuminada(A3)]

# ╔═╡ 18db005d-d530-43ca-bc88-d00d0b39d8ca
md"""
Es evidente que los resultados varian dependiendo del tipo de ruido que hay en la imagen. Los reultados son relativamente satisfactorios para la primera y la tercera imagen, aunque se pierden ciertos aspectos de la imagen original. Para la segunda imagen, los resultados no son tan satisfactorios; ello se debe a la tendencia del promedio a verse afectado por los outliers (recuerde que en el ruido sal y pimienta, los pixeles afectados presentan valores extremos). 
"""

# ╔═╡ 28b1d1ac-1610-4a7c-b22d-9e66f1c8756b
md"""##   $\cdot$ Difuminado con pesos"""

# ╔═╡ 092b1c64-fbab-4d3a-afde-9c5220fcddac
md"""
Otro enfoque comúnmente utilizado es realizar un promedio ponderado con pesos sobre la vecindad de los pixeles. La elección de estos pesos puede depender, por ejemplo, de una función. A continuación, la *familia gaussiana de pesos*. Considere la función de densidad de una distribución gaussiana bivariada con variables independientes

$\large g(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2 \sigma^2}\right)$
"""

# ╔═╡ cdc2a866-d7cc-4044-93b1-14828a18f527
md"""
Si quisiéramos difuminar una imagen con los valores que se presentan en una vecindad de tamaño $3\times 3$ de cada pixel, identificamos el pixel que se quiere transformar con el centro coordenado $x=y=0$, y la vecindad como se presenta a continuación.
"""

# ╔═╡ 410b6e39-bafd-4007-bdf1-29725bab4f62
md"""

$

\begin{array}{|c|c|c|}
\hline \\
\hspace{0.5cm}(1,-1)\hspace{0.5cm} & \hspace{0.5cm}(1,0)\hspace{0.5cm} & \hspace{0.5cm}(1,1)\hspace{0.5cm}  \\ \\
\hline \\
\hspace{0.5cm}(0,-1)\hspace{0.5cm} & \hspace{0.5cm}(0,0)\hspace{0.5cm} & \hspace{0.5cm}(0,1)\hspace{0.5cm} \\ \\ 
\hline \\
\hspace{0.5cm}(-1,-1)\hspace{0.5cm} & \hspace{0.5cm}(-1,0)\hspace{0.5cm} & \hspace{0.5cm}(-1,1)\hspace{0.5cm} \\ \\ 
\hline
\end{array}$
"""


# ╔═╡ cdd2a6f8-8e9f-40c4-90fd-f7f73b8091e3
md"""
Con $\sigma = 1$ evaluamos la funcion $g(x,y)$ en cada elemento de esta vecindad para obtener la matriz de pesos.
"""

# ╔═╡ 4ad86246-2890-43e1-a646-609e95fb47e7
md"""

$\large \begin{bmatrix}0.0585 & 0.0965 & 0.0585 \\
0.0965 & 0.1592 & 0.0965 \\
0.0585 & 0.0965 & 0.0585 \\ \end{bmatrix}$

"""

# ╔═╡ c242c040-224f-4d2f-86a7-7e8bb98c0e7f
md"""
Vemos que $2* 0.0293 \approx 0.0585$, $3*0.0293 = 0.0878 \approx 0.0965$ y $5*0.0293 = 0.1464 \approx 0.1592$. De esta forma, y aproximando de manera conveniente tal que la suma de todas las entradas en la matriz den $1$, obtenemos una matriz aproximada de pesos

$\LARGE \begin{bmatrix}\frac{2}{25} & \frac{3}{25} & \frac{2}{25} \\
\frac{3}{25} & \frac{5}{25} & \frac{3}{25} \\
\frac{2}{25} & \frac{3}{25} & \frac{2}{25} \\ \end{bmatrix}$

En este método, tanto el tamaño de la vecindad como la varianza de la distribución normal puede variar. Ejemplos de la utilización de este procedimiento (con una matriz de pesos $3\times 3$ y $\sigma=1$) se presentan a continuación.
"""

# ╔═╡ eda71d34-810a-4ce1-bc6b-3040651d7ec1
function imagen_difuminada_pesos(A)
    A = Float64.(channelview(A))
    M, N = size(A)
    for m in 2:M-1
        for n in 2:N-1
            A[m, n] = 2/25*A[m-1, n-1] + 3/25*A[m, n-1] + 2/25*A[m+1, n-1] +
                       3/25*A[m-1, n] + 5/25*A[m, n] + 3/25*A[m+1, n] +
                       2/25*A[m-1, n+1] + 3/25*A[m, n+1] + 2/25*A[m+1, n+1]
        end
    end
	return Gray.(A)
end

# ╔═╡ c15788f6-9a9f-49c5-8f3f-3efcb7d5e76c
[A imagen_difuminada(A) imagen_difuminada_pesos(A)]

# ╔═╡ d813e58c-da2e-48d2-b81c-6c508f4d22db
[A1 imagen_difuminada_pesos(A1)]

# ╔═╡ 099b256d-ae64-4426-ab9c-166e45ce9cb2
[A2 imagen_difuminada_pesos(A2)]

# ╔═╡ 80df2389-a06f-4be7-b956-6da33b84fb36
[A3 imagen_difuminada_pesos(A3)]

# ╔═╡ 23dde8c5-ff59-48b7-b608-d48fb45001c1
md"""
Note cómo en la primera imagen también se reduce el ruido y, de cierta manera, aumenta la nitidez ligeralmente. Al procedimiento de reemplazar un pixel de la imagen con un promedio ponderado de su vecindad se le conoce como *convolución*. La matriz de pesos que se utiliza para realizar tal promedio es denominada *filtro* o *matriz de convolución*, y aplicar el procedimiento de *convolución* sobre una imagen con ruido bajo cierto *filtro* es denominado *filtrado*. Por ejemplo, en las anteriores imágenes, hemos aplicado filtrado de las imágenes con distintos ruidos usando los filtros

$\begin{bmatrix}\frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\
\frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\
\frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\ \end{bmatrix} \textup{ y }\begin{bmatrix}\frac{2}{25} & \frac{3}{25} & \frac{2}{25} \\
\frac{3}{25} & \frac{5}{25} & \frac{3}{25} \\
\frac{2}{25} & \frac{3}{25} & \frac{2}{25} \\ \end{bmatrix}$

Al primero se le denomina *difuminado de promedio*, mientras que al segundo procedimiento, con cualquier tamaño de vecindad y cualquier desviación estándar, se le denomina *difuminado gaussiano*.

"""

# ╔═╡ 8bb59b12-5978-4b11-9c56-a2dbcf049f94
md"""
## Filtrado de mediana
"""

# ╔═╡ 434a8b4e-42a7-41c2-82a4-2e9ef36164c2
md"""
Un método que suele ser eficiente para reducir el ruido sal y pimienta es el ruido de la mediana. Como se mencionó anteriormente, al ser este un filtro que presenta "outliers", el promedio no parece ser la herramienta más adecuada para la eliminación del ruido, Utilizaremos la mediana. Para ello, transformaremos cada pixel en la mediana de los valores en su vecindad, que, en este ejemplo, tomaremos como $3\times 3$.
"""

# ╔═╡ 53f72643-39db-4202-ae6e-fff79e4539b3
function median_filtering(A)
	A = Float64.(channelview(A))
    M, N = size(A)
	for m in 2:M-1
        for n in 2:N-1
			x=[]
            for i in -1:1
				for j in -1:1
					append!(x,A[m+i,n+j])
				end
			end
			x=sort(x)
			A[m,n]=x[5]
        end
    end
	return Gray.(A)
end

# ╔═╡ fad9d583-58f3-4ec4-b4df-f4fd4d0e48be
[A2 median_filtering(A2)]

# ╔═╡ edfcd89d-3a7d-4f4e-b6c3-6534bd6854e3
md"""
## Fitrado de Variación Total
"""

# ╔═╡ bde84449-68fe-4069-8734-b406e29e2393
md"""
Otra opción es intentar traducir nuestro problema de filtrado en un problema de optimización. A partir de una imagen $B$ con ruido, quisieramos hallar un vector de $A$ que 

- Se parezca lo que más se pueda a la imagen con ruido $B$
- Tenga 'variación total' suave. Es decir, se minimice la diferencia entre los valores de un pixel y aquellos de su vecindad. 

Es decir, quisiéramos resolver el problema 

$\min_{A\in \mathbb{R}^{M\times N}} d(A,B)+\lambda v(A),$

donde $d(\cdot,\cdot)$ es una medida de distancia, $v(A)$ es una medida de la variación entre los pixeles, y $\lambda$ es un parámetro que mide la importancia que le damos a cada uno de estos factores (note que el problema con $\lambda=0$ tiene solución la matriz con ruido). Por ejemplo, podemos considerar el problema

$\min_{A\in \mathbb{R}^{M\times N}} \sum_{i=1}^{N}\sum_{j=1}^{k}(B[i,j]-A[i,j])^2+\lambda \left(\sum_{i=1}^{N-1}\sum_{j=1}^{N-1} \left|A[i,j]-A[i,j+1]\right|+|A[i,j]-A[i+1,j]|\right)$

Se han desarrollado varios métodos que intentan resolver este problema de manera eficiente, incluyendo métodos variacionales que pretenden solucionar el problema variacional equivalente (reducir infinitesimalmente los pixeles de la imagen). No obstante, se escapan del alcance de este cuaderno, e intentaremos resolver este problema mediante optimización numérica.
"""

# ╔═╡ fc3bc625-fd58-47a5-9c49-58fbf2b1eab8
[A1 total_variation(A1,1.5)]

# ╔═╡ b23694ab-3556-4d78-a95b-7a9d7b27c521
[A2 total_variation(A2,1.2)]

# ╔═╡ ba8214a4-b75e-4e0d-bd82-e4a57f030d02
[A3 total_variation(A3,1.5)]

# ╔═╡ f6304aa9-2385-4ca8-9404-97c371553a2b
md"""
Nótese que un parámetro $\lambda$ muy grande resulta en una imagen excesivamente borrosa, como se muestra a continuación.
"""

# ╔═╡ a175a1c9-32fa-4493-b7cb-4c630d51770d
[A1 total_variation(A1,3)]

# ╔═╡ ac4fd8ff-a4cd-4c5d-946c-931af96d540b
md"""
## Otra imagen
"""

# ╔═╡ 05f8d1c7-689d-4267-b266-65571667fe7e
begin
	url1="https://github.com/ytrujillol/Procesamiento-de-imagenes/blob/main/Images/Eagle.jpg?raw=true"
	f1name = download(url1)
	T = Gray.(load(f1name));
end

# ╔═╡ a6454a40-9f73-48e7-a83b-be5c6e33f70e
md"""
## Difuminado
"""

# ╔═╡ 3a34b778-cbbc-4c7c-adad-bbb2a73c3eb8
[T imagen_difuminada(T)]

# ╔═╡ 2a84219b-8f2e-4934-9831-b1484dec8de9
T1 = ruido_gaussiano(T,0.2);

# ╔═╡ 1a0ad46a-fa61-4191-87cc-f20ed6ba0bc8
T2 = sal_y_pimienta(T,0.1);

# ╔═╡ bf1e0a76-b741-43d4-aa30-6c67d3023062
T3 = agregar_ruido_speckle(T, 0.3);

# ╔═╡ 04778277-36c3-4e5b-b7e4-a2519631fe2a
md"""
## Filtrado

Se aplican los distintos filtros presentados para la imagen con ruido.
"""

# ╔═╡ 73112f06-fdb8-4523-8f46-eddf36da04b2
[T1 imagen_difuminada(T1) imagen_difuminada_pesos(T1) median_filtering(T1) total_variation(T1,1)]

# ╔═╡ 5fed002f-0a43-4fb1-99a4-fb18aa9674ea
[T2 imagen_difuminada(T2) imagen_difuminada_pesos(T2) median_filtering(T2) total_variation(T2,1)]

# ╔═╡ 490055bf-cf98-4e51-81ea-822c8db1a9ce
[T3 imagen_difuminada(T3) imagen_difuminada_pesos(T3) median_filtering(T3) total_variation(T3,1)]

# ╔═╡ Cell order:
# ╟─d9d4169d-0af8-43e5-9611-7cf8ff69e5cd
# ╟─e0fcabe0-b120-11ef-1869-cf0f1342a959
# ╠═402b40b3-ad31-4561-8880-adb973772f32
# ╟─6c455d3d-f1e5-4c40-8724-a18d30166f36
# ╟─899b6b24-4e39-491c-b4b2-b4484cb5e2d4
# ╟─8593931b-f918-4395-a600-ef51e151dd32
# ╟─3cda7462-9628-4006-b39d-f1cfbf61713e
# ╟─4c606d57-c4a8-465b-a043-f89ee4f491b7
# ╟─f4151cb3-8097-4da0-9f2d-a52118e23e37
# ╟─f2b3bbd9-2bd6-4d48-86db-93fbf736a20c
# ╟─41fa8758-47ef-400a-97dd-2e45fda34391
# ╟─0d8d6355-460e-404d-b6ce-713c0c6ed713
# ╠═9afafb0f-ac0b-470b-aab3-b41e0a0dc0f3
# ╠═11076927-b844-4261-8069-6e183f60844c
# ╟─7f82bbae-a8ae-46a8-8fc3-92dd23079590
# ╟─54390a0b-7335-4437-8e48-ee3391ca15b4
# ╠═de95337f-073d-4cab-ad17-ac8e90192de4
# ╠═1beec73a-f89d-4c68-9642-887a064e983a
# ╟─b8b84a35-4b7d-47be-ac66-0b2d490d5b65
# ╟─3dfcbe46-a26a-484e-ad2d-f25dca23716b
# ╠═d163328a-74cc-453c-9dbc-ee3b14ced35a
# ╠═d02b4097-e9be-40f4-a52b-dc54ecd1fb2c
# ╟─cb8d7d58-d0f8-43ed-a168-a292d7880f6d
# ╟─340808ee-0131-4c33-b17e-2b35067f93ae
# ╠═cc5d32c4-b124-4a5b-af7f-a69be2c309e1
# ╟─4943534b-e2a0-4733-b630-0f743df58f6f
# ╟─417e745f-6380-472c-bfe2-18bb92f42a5f
# ╠═9bfdce84-9554-41cf-a042-97d730c5471c
# ╠═d34c0c51-e87a-4732-bea5-ceb01a74aefa
# ╠═e9a3d6ae-52b0-4949-8985-5cb3f890474a
# ╟─226c10a9-2cd1-4683-8aff-38db6d54dc72
# ╟─3aa265d0-44d7-4e4b-aae2-b7b9d628d04b
# ╟─aae5b170-5979-4b32-b92a-f0e5e563aa29
# ╟─18db005d-d530-43ca-bc88-d00d0b39d8ca
# ╟─28b1d1ac-1610-4a7c-b22d-9e66f1c8756b
# ╟─092b1c64-fbab-4d3a-afde-9c5220fcddac
# ╟─cdc2a866-d7cc-4044-93b1-14828a18f527
# ╟─410b6e39-bafd-4007-bdf1-29725bab4f62
# ╟─cdd2a6f8-8e9f-40c4-90fd-f7f73b8091e3
# ╟─4ad86246-2890-43e1-a646-609e95fb47e7
# ╟─c242c040-224f-4d2f-86a7-7e8bb98c0e7f
# ╠═eda71d34-810a-4ce1-bc6b-3040651d7ec1
# ╠═c15788f6-9a9f-49c5-8f3f-3efcb7d5e76c
# ╠═d813e58c-da2e-48d2-b81c-6c508f4d22db
# ╠═099b256d-ae64-4426-ab9c-166e45ce9cb2
# ╠═80df2389-a06f-4be7-b956-6da33b84fb36
# ╟─23dde8c5-ff59-48b7-b608-d48fb45001c1
# ╟─8bb59b12-5978-4b11-9c56-a2dbcf049f94
# ╟─434a8b4e-42a7-41c2-82a4-2e9ef36164c2
# ╠═53f72643-39db-4202-ae6e-fff79e4539b3
# ╠═fad9d583-58f3-4ec4-b4df-f4fd4d0e48be
# ╟─edfcd89d-3a7d-4f4e-b6c3-6534bd6854e3
# ╟─bde84449-68fe-4069-8734-b406e29e2393
# ╠═1e758897-b3f6-4813-812c-70f2ee88a3b8
# ╠═fc3bc625-fd58-47a5-9c49-58fbf2b1eab8
# ╠═b23694ab-3556-4d78-a95b-7a9d7b27c521
# ╠═ba8214a4-b75e-4e0d-bd82-e4a57f030d02
# ╟─f6304aa9-2385-4ca8-9404-97c371553a2b
# ╠═a175a1c9-32fa-4493-b7cb-4c630d51770d
# ╟─ac4fd8ff-a4cd-4c5d-946c-931af96d540b
# ╟─05f8d1c7-689d-4267-b266-65571667fe7e
# ╟─a6454a40-9f73-48e7-a83b-be5c6e33f70e
# ╠═3a34b778-cbbc-4c7c-adad-bbb2a73c3eb8
# ╠═2a84219b-8f2e-4934-9831-b1484dec8de9
# ╠═1a0ad46a-fa61-4191-87cc-f20ed6ba0bc8
# ╠═bf1e0a76-b741-43d4-aa30-6c67d3023062
# ╟─04778277-36c3-4e5b-b7e4-a2519631fe2a
# ╠═73112f06-fdb8-4523-8f46-eddf36da04b2
# ╠═5fed002f-0a43-4fb1-99a4-fb18aa9674ea
# ╠═490055bf-cf98-4e51-81ea-822c8db1a9ce
