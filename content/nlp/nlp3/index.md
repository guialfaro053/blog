---
title: "Yet another NVFP4 tutorial" 
# date: 2023-09-07
weight: 3
tags: ["NVFP4","Blackwell"]
author: ["Guillermo Alfaro. October 2026"]
description: ""
summary: "For the past few weeks, I've been trying to understand how and why NVFP4 works and writing some kernels on my Blackwell machine (RTX 5070 Ti), and I wanted to share a bit of how I understand it."
cover:
    image: "nvfp4.png"
    alt: "NVFP4"
    relative: True
editPost:
    URL: "https://gist.github.com/guialfaro053/feec6f6e0724e995428b78af1e5bbfeb"
    Text: "NVFP4 GitHub gist"
showToc: true
disableAnchoredHeadings: true

---

For the past few weeks, I've been trying to understand how and why NVFP4 works and writing some kernels on my Blackwell machine (RTX 5070 Ti), and I wanted to share a bit of how I understand it.

## What is the NVFP4 format?

Simply said, NVFP4 is the 4-bit floating point format Blackwell GPUs can multiply natively, but before we come back to this, let's try to understand why this is even needed and why it is such a tough task.

Quantization, in the context of model weights, is what we use to express high precision values as lower precision values, willing to sacrifice some accuracy in exchange for using fewer resources. A quick mental exercise. A format with $b$ bits has $2^b$ possible bit patterns, so fp32 has $2^{32}$ (around 4 billion), fp16 around $2^{16} \approx 65$k, fp8 $2^8 = 256$, and fp4 only $2^4 = 16$. How each format splits those bits into sign, exponent and mantissa decides which values they represent (IEEE 754 defines this for fp32 and fp16, while fp8 and fp4 come from the OCP Microscaling spec). So the task is to represent what fp32 can in just 16 possibilities, reducing the memory footprint up to $8\times$ (or $4\times$ if we start from fp16 or bf16).

In order to tackle this, we are presented with a very ingenious algorithm, [the NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). NVFP4 packs tensors into blocks of 16 elements and creates a scale for each block in fp8 (E4M3). On top of that, the whole tensor gets one more scale, which we call the tensor scale, stored in fp32 and affecting the whole tensor equally. The block scale is chosen so the largest value in the block maps to $6$, the largest value E2M1 can represent. After applying both scales, every element lands somewhere in $[-6, 6]$ and gets rounded to its nearest E2M1 value (in PyTorch you can emulate this with `torch.clamp` for the range and `torch.bucketize` for the rounding). Getting the value back is just multiplying everything again

$$
x \approx s_{\text{tensor}} \cdot s_{\text{block}} \cdot q
$$

where $q$ is the 4-bit E2M1 value, $s_{\text{block}}$ is the fp8 scale of the block $x$ belongs to, and $s_{\text{tensor}}$ is the fp32 scale shared by the whole tensor.

One of the advantages of granularizing as much as this algorithm does is that it becomes incredibly robust to outliers, which are famous for being an issue in quantization. An outlier can only stretch the scale of its own 16-element block instead of ruining the precision of the entire tensor.

If we try to measure the gains of storing an fp32 tensor of shape $m \times n$ in the NVFP4 format, we have to count the 4-bit elements, one 8-bit scale per block of 16 elements and a single 32-bit tensor scale. Dividing the total bits by the number of elements gives us the average cost per element

$$
\lim_{m,n \to \infty} \frac{4mn + 8 \cdot \frac{mn}{16} + 32}{mn} = \lim_{m,n \to \infty} \left( 4 + \frac{1}{2} + \frac{32}{mn} \right) = 4.5 \text{ bits per element}
$$

Here $mn$ is the number of elements in the tensor, and $4mn$ is the cost of storing each of them as a 4-bit E2M1 value. There are $\frac{mn}{16}$ blocks with an 8-bit scale each, so $8 \cdot \frac{mn}{16} = \frac{mn}{2}$ adds half a bit per element. The $32$ is the single fp32 tensor scale, a constant that doesn't grow with the tensor, so as $m$ and $n$ get large, $\frac{32}{mn}$ vanishes and we are left with 4.5 bits per element.

Compared to the 32 bits of fp32, this gives us a compression ratio of

$$
\lim_{m,n \to \infty} \frac{32mn}{4.5mn + 32} = \frac{32}{4.5} \approx 7.1\times
$$

or $\frac{16}{4.5} \approx 3.6\times$ compared to fp16 or bf16.


## The 4-bit codes

For a 4-bit quantization, we can map the representations from `0000` through `1111` (or `0x0` through `0xF` in hex). NVFP4 stores its elements as E2M1, which means 1 sign bit $S$, 2 exponent bits $E$ and 1 mantissa bit $M$, with an exponent bias of 1. The value of each code is given by

$$
v = (-1)^S \cdot 2^{E-1} \cdot \left( 1 + \frac{M}{2} \right) \qquad \text{if } E > 0 \text{ (normal)}
$$

$$
v = (-1)^S \cdot \frac{M}{2} \qquad \text{if } E = 0 \text{ (subnormal)}
$$

which gives us the following table.

<div class="center-table">

| Hex | Binary | $S$ | $E$ | $M$ | Value |
|:---:|:------:|:---:|:---:|:---:|:-----:|
| `0x0` | `0000` | 0 | 00 | 0 | $0$ |
| `0x1` | `0001` | 0 | 00 | 1 | $0.5$ |
| `0x2` | `0010` | 0 | 01 | 0 | $1$ |
| `0x3` | `0011` | 0 | 01 | 1 | $1.5$ |
| `0x4` | `0100` | 0 | 10 | 0 | $2$ |
| `0x5` | `0101` | 0 | 10 | 1 | $3$ |
| `0x6` | `0110` | 0 | 11 | 0 | $4$ |
| `0x7` | `0111` | 0 | 11 | 1 | $6$ |
| `0x8` | `1000` | 1 | 00 | 0 | $-0$ |
| `0x9` | `1001` | 1 | 00 | 1 | $-0.5$ |
| `0xA` | `1010` | 1 | 01 | 0 | $-1$ |
| `0xB` | `1011` | 1 | 01 | 1 | $-1.5$ |
| `0xC` | `1100` | 1 | 10 | 0 | $-2$ |
| `0xD` | `1101` | 1 | 10 | 1 | $-3$ |
| `0xE` | `1110` | 1 | 11 | 0 | $-4$ |
| `0xF` | `1111` | 1 | 11 | 1 | $-6$ |

</div>

One thing I would like to mention is that $-0$ and $0$ are the same number, but they have two different codes (`0x0` and `0x8`). If we are approximating a value to its nearest E2M1 representable value, we can land at zero from the left ($-0.1$ becomes $-0$) or from the right ($0.1$ becomes $0$). When we dequantize, this isn't an issue since both give back zero no matter the scale, but it means one of our 16 codes is wasted and we only get 15 distinct values. It also means that if you compare the raw 4-bit codes in a kernel instead of their decoded values, `0x0` and `0x8` won't be equal even though both mean zero.

## Quantizing a tensor

Let's do a walkthrough of the quantization process on paper. The code is at the end.

Let's use one row of 32 weight-like values, which makes two blocks of 16. Block 0 holds some larger values and block 1 holds small ones so we can see why each block gets its own scale.

```
block 0: 0.12 -0.05  0.31  0.02 -0.18  0.07  0.00 -0.09  0.15 -0.22  0.04  0.11 -0.01  0.26 -0.13  0.08
block 1: 0.010 -0.030 0.045 0.005 -0.020 0.015 -0.040 0.025 0.000 0.035 -0.008 0.012 -0.050 0.018 0.003 -0.027
```

Quantizing turns each value $x_i$ into a 4-bit code, which shares an fp8 block scale with the other 15 elements of its block and one fp32 tensor scale with the whole tensor. Decoding reverses it

$$
x_i \approx \frac{q_i \cdot s_{\text{block}}}{s_{\text{enc}}}
$$

where $q_i$ is the E2M1 value of the code. This is the same formula as before with $s_{\text{tensor}} = \frac{1}{s_{\text{enc}}}$.

### Step 1, the tensor scale (one per tensor)

$$
\text{amax} = \max_i \lvert x_i \rvert = 0.31 \qquad s_{\text{enc}} = \frac{448 \cdot 6}{\text{amax}} = \frac{2688}{0.31} \approx 8670.97
$$

Why $448 \cdot 6$? $448$ is the largest E4M3 block scale and $6$ is the largest E2M1 value, so $2688$ is the biggest number the two layers can represent together. Multiplying everything by $s_{\text{enc}}$ stretches the tensor so its largest value lands exactly at $2688$, which uses the full range of the block scales.

Block 0 after scaling looks like this

```
1040.5  -433.5  2688.0  173.4  -1560.8  607.0  0.0  -780.4  1300.6  -1907.6  346.8  953.8  -86.7  2254.5  -1127.2  693.7
```

### Step 2, the block scale (one per 16 elements)

Each block asks "what scale makes my largest value land at $6$?"

$$
s_{\text{block}} = \text{E4M3}\left( \frac{\max_i \lvert s_{\text{enc}} \cdot x_i \rvert}{6} \right)
$$

<div class="center-table">

| Block | Scaled amax | $\div 6$ | Rounded to E4M3 | Stored byte |
|:-----:|:-----------:|:--------:|:---------------:|:-----------:|
| 0 | $2688.0$ | $448.0$ | $448$ | `0x7E` |
| 1 | $433.5$ | $72.26$ | $72$ | `0x69` |

</div>

The block scale is itself an 8-bit float (E4M3), so it gets rounded too. $72.26$ isn't representable, so it becomes $72$, and that small rounding comes back in step 3. With a bias of 7, the byte `0x69` is `0 1101 001`, which decodes as

$$
2^{13 - 7} \cdot \left( 1 + \frac{1}{8} \right) = 2^6 \cdot 1.125 = 72
$$

This step is the core idea of tackling outliers. Block 1's values are about $6\times$ smaller than block 0's, but they get their own smaller scale, so they still use the whole range from $-6$ to $6$. If we had only used one scale for the whole tensor, most of block 1 would round to $0$ or $\pm 0.5$.

### Step 3, divide by the block scale and round to the E2M1 grid

Here we divide each scaled value by its block's scale, clamp it to $[-6, 6]$ and snap it to the nearest of $0, 0.5, 1, 1.5, 2, 3, 4, 6$ (keeping the sign)

$$
q_i = \text{E2M1}\left( \text{clamp}\left( \frac{s_{\text{enc}} \cdot x_i}{s_{\text{block}}}, -6, 6 \right) \right)
$$

For block 0 we divide by $448$

```
ratio:  2.32 -0.97  6.00  0.39 -3.48  1.35  0.00 -1.74  2.90 -4.26  0.77  2.13 -0.19  5.03 -2.52  1.55
q:      2    -1     6     0.5  -3     1.5   0    -1.5   3    -4     1     2    -0     6    -3     1.5
```

and for block 1 we divide by $72$

```
ratio:  1.20 -3.61  5.42  0.60 -2.41  1.81 -4.82  3.01  0.00  4.22 -0.96  1.45 -6.02  2.17  0.36 -3.25
q:      1    -4     6     0.5  -2     2    -4     3     0     4    -1     1.5  -6     2     0.5  -3
```

Now, let's look at a couple of interesting things that happened while rounding.

- $-6.02$ gets clamped to $-6$. This comes from step 2, since rounding the scale down from $72.26$ to $72$ pushed this element slightly past $6$.
- $5.03$ becomes $6$. The midpoint between $4$ and $6$ is $5$, and $5.03$ is just above it. This is the largest error in the example values since $0.26$ decodes back as $0.31$.
- $-0.19$ becomes $-0$. Small values fall to zero, so the original $-0.01$ is lost.

This step is the only one where we actually lose information. Everything after it is exact bookkeeping.

### Step 4, values to 4-bit codes

Each $q_i$ becomes a nibble $c_i = 8S + j$, where the sign $S$ is bit 3 and the low 3 bits $j$ are the position of $\lvert q_i \rvert$ in $(0, 0.5, 1, 1.5, 2, 3, 4, 6)$. For block 0 we get

```
q:     2    -1   6    0.5  -3   1.5  0   -1.5  3    -4   1    2    -0   6    -3   1.5
code:  0x4  0xA  0x7  0x1  0xD  0x3  0x0  0xB  0x5  0xE  0x2  0x4  0x8  0x7  0xD  0x3
```

For instance, for $-3$ the magnitude is at index $5$ (`101`), and the sign bit adds $8$, giving `1101` = `0xD`. Also note that $-0$ became `0x8` (sign bit set, magnitude $0$), which decodes to zero either way.

### Step 5, pack two codes per byte

Two 4-bit codes fit in one byte, with the even element in the low nibble and the odd element in the high nibble

$$
b_k = c_{2k} + 16 \cdot c_{2k+1}
$$

which in PyTorch is `packed = codes[:, 0::2] | (codes[:, 1::2] << 4)`.

```
pair:    (0x4,0xA) (0x7,0x1) (0xD,0x3) (0x0,0xB) (0x5,0xE) (0x2,0x4) (0x8,0x7) (0xD,0x3)
byte:      0xA4      0x17      0x3D      0xB0      0xE5      0x42      0x78      0x3D
```

Reading each byte right to left gives the elements in order, so `0xA4` means element 0 is `4` (value $2$) and element 1 is `A` (value $-1$).

### What we get at the end

```
codes        : A4 17 3D B0 E5 42 78 3D  E2 17 4C 5E 60 3A 4F D1   (16 bytes)
block_scales : 7E 69                                              (2 bytes)
global_scale : 8670.97                                            (4 bytes, once per tensor)
```

Here `global_scale` is $s_{\text{enc}}$, so the tensor scale from the first section is its inverse, $s_{\text{tensor}} = \frac{1}{s_{\text{enc}}}$.

## Matmuls

The whole point of this adventure was not only to store data in lower-precision values but to also do operations on it. One of the bottlenecks when working on GPUs is moving data from one place to another within the GPU. Since we are storing the same information in less space, it's faster to move it around, which gives NVFP4 a speed boost when the workload is memory bound. And if the hardware allows, the fp4 tensor cores can do more multiply-adds per cycle than fp16.

I have written the code for conversion and matmuls in torch and triton. On hardware with fp4 tensor cores the core of the matmul is a single line:

```python
tl.dot_scaled(a, sa, "e2m1", w.T, sw, "e2m1", acc)
```

For each operand, the arguments are the packed tile, its block scales and its format "e2m1". Code can be found [here](https://gist.github.com/guialfaro053/feec6f6e0724e995428b78af1e5bbfeb).
