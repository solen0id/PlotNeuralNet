import os


def make_header(projectpath):
    pathlayers = os.path.join(projectpath, "layers/").replace("\\", "/")
    return (
        r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""
        + pathlayers
        + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
"""
    )


def make_colors():
    return r"""
\def\InputColor{rgb:gray,8}
\def\OutputColor{rgb:gray,3}
\def\ConvColor{rgb:orange,7;yellow,2;white,1}
\def\ConvStridedColor{rgb:orange,8;yellow,1;white,1}
\def\PoolColor{rgb:orange,9;red,3;black,4}
\def\DeconvColor{rgb:blue,7;cyan,2;white,3}
\def\DeconvStridedColor{rgb:blue,8;cyan,1;white,1}
\def\UnpoolColor{rgb:blue,6;cyan,3;black,3}
\def\SoftmaxActivationColor{rgb:green,1;yellow,9;white,1}
\def\SigmoidActivationColor{rgb:green,6;lime,3;white,1}
\def\LeakyReluActivationColor{rgb:green,6;lime,3;white,1}
\def\TanhActivationColor{rgb:green,1;yellow,9;white,1}
\def\FlatColor{rgb:purple,7;magenta,3;white,2}
\def\ReshapeColor{rgb:purple,6;magenta,2;black,5}
\def\FcColor{rgb:red,7;pink,2;white,1}
\def\BatchNormColor{rgb:gray,5;white,5}
"""


def begin_document():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""


def create_legend(
    input_=False,
    conv=False,
    conv_strided=False,
    deconv=False,
    deconv_strided=False,
    pool=False,
    unpool=False,
    fc=False,
    flatten=False,
    reshape=False,
    softmax=False,
    sigmoid=False,
    leaky_relu=False,
    tanh=False,
    batch_norm=False,
):
    layers = []

    if input_:
        layers.append((r"\InputColor", "Input"))
    if conv:
        layers.append((r"\ConvColor", "Convolution"))
    if conv_strided:
        layers.append((r"\ConvStridedColor", "Strided\\Convolution"))
    if deconv:
        layers.append((r"\DeconvColor", "Transposed\\Convolution"))
    if deconv_strided:
        layers.append((r"\DeconvStridedColor", "Strided\\Transposed\\Convolution"))
    if pool:
        layers.append((r"\PoolColor", "Down-\\sampling"))
    if unpool:
        layers.append((r"\UnpoolColor", "Up-\\sampling"))
    if fc:
        layers.append((r"\FcColor", "Fully\\Connected"))
    if flatten:
        layers.append((r"\FlatColor", "Flatten"))
    if reshape:
        layers.append((r"\ReshapeColor", "Reshape"))
    if softmax:
        layers.append((r"\SoftmaxActivationColor", "Softmax \\ Activation"))
    if sigmoid:
        layers.append((r"\SigmoidActivationColor", "Sigmoid \\ Activation"))
    if leaky_relu:
        layers.append((r"\LeakyReluActivationColor", "Leaky ReLU \\ Activation"))
    if tanh:
        layers.append((r"\TanhActivationColor", "Tanh \\ Activation"))
    if batch_norm:
        layers.append((r"\BatchNormColor", "Batch\\Normalization"))

    if not layers:
        return "\n"

    color_squares = " & ".join(
        [
            rf"\tikz\draw[fill={color}, opacity=0.4] (0,0) rectangle (0.5,0.5);"
            for color, _ in layers
        ]
    )
    titles = " & ".join(
        [
            rf"\begin{{tabular}}{{c}} {title.replace('\\', ' \\\\ ')} \end{{tabular}}"
            for _, title in layers
        ]
    )

    return rf"""
\node[anchor=north west, align=center, draw=black, fill=white, rounded corners, minimum width=\textwidth, inner sep=5pt]
    at ([yshift=-1cm]current bounding box.south west) {{
    \begin{{tabular}}{{{'c' * len(layers)}}}
    {color_squares} \\
    {titles} \\
    \end{{tabular}}
}};
"""


# layers definition
# fmt: off
def genericLayer(name, fill, xlabel="", zlabel="", offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    xlabel = f"{xlabel}" if xlabel not in ("", None) else ""
    zlabel = f"{zlabel}" if zlabel  not in ("", None) else ""

    return r'''
\pic[shift={''' + str(offset) + r'''}] at ''' + origin + r'''
    {Box={
        name=''' + name + r''',
        caption={''' + caption + r'''},
        xlabel={''' + xlabel + r'''},
        zlabel={''' + zlabel + r'''},
        fill=''' + str(fill) + r''',
        height=''' + str(height) + r''',
        depth=''' + str(depth) + r''',
        width=''' + str(width) + r'''
        }
    };
    '''

def genericConvLayer(name, num_channels, size, fill=r"\ConvColor", offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    # x_label = num_channels if num_channels is not None else ""
    z_label = f"{size[0]}x{size[1]}" if size is not None else ""
    z_label = f"{z_label}x{num_channels}" if num_channels is not None else z_label
    return genericLayer(name, fill, "", z_label, offset, origin, width, height, depth, caption)

def genericFlatLayer(name, num_units, fill=r"\FlatColor", offset="(0,0,0)", origin="(0,0,0)", width=1, height=10, depth=10, caption=" "):
    x_label = num_units if num_units is not None else ""
    return genericLayer(name, fill, x_label, "", offset, origin, width, height, depth, caption)

def inputLayer(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\InputColor", offset, origin, width, height, depth, caption)

def outputLayer(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\OutputColor", offset, origin, width, height, depth, caption)

def conv2D(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\ConvColor", offset, origin, width, height, depth, caption)

def conv2D_strided(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\ConvStridedColor", offset, origin, width, height, depth, caption)

def deconv2D(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\DeconvColor", offset, origin, width, height, depth, caption)

def deconv2D_strided(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\DeconvStridedColor", offset, origin, width, height, depth, caption)

def downsample(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size,r"\PoolColor", offset, origin, width, height, depth, caption)

def upsample(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\UnpoolColor", offset, origin, width, height, depth, caption)

def fully_connected(name, num_units, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericFlatLayer(name, num_units, r"\FcColor", offset, origin, width, height, depth, caption)

def flatten(name, num_units, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericFlatLayer(name, num_units, r"\FlatColor", offset, origin, width, height, depth, caption)

def reshape(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\ReshapeColor", offset, origin, width, height, depth, caption)

def softmax(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\SoftmaxActivationColor", offset, origin, width, height, depth, caption)

def leaky_relu(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)",  width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\LeakyReluActivationColor",  offset, origin, width, height, depth, caption)

def tanh(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\TanhActivationColor", offset, origin, width, height, depth, caption)

def batch_norm(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return genericConvLayer(name, num_channels, size, r"\BatchNormColor", offset, origin, width, height, depth, caption)


def conv_lrelu_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return (
        conv2D(f"{name}_in",  None, None, offset, origin, width, height, depth, caption="")
        + leaky_relu(f"{name}_out", num_channels, size,offset="(0,0,0)", origin=f"({name}_in-east)", width=1, height=height, depth=depth, caption=caption)
    )

def strided_conv_lrelu_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return (
        conv2D_strided(f"{name}_in",  None, None, offset, origin, width, height, depth, caption="")
        + leaky_relu(f"{name}_out",num_channels, size, offset="(0,0,0)", origin=f"({name}_in-east)", width=1, height=height, depth=depth, caption=caption)
    )

def conv_lrelu_pool_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return (
        conv2D(f"{name}_in",  None, None, offset, origin, width, height+5, depth+5, caption="")
        + leaky_relu(f"{name}_lrelu",  None, None, offset="(0,0,0)", origin=f"({name}_in-east)", width=1, height=height+5, depth=depth+5, caption="")
        + downsample(f"{name}_out", num_channels, size, offset="(0,0,0)", origin=f"({name}_lrelu-east)", width=width, height=height, depth=depth, caption=caption)
    )

def conv_lrelu_batchnorm_pool_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return (
        conv2D(f"{name}_in",  None, None, offset, origin, width, height+5, depth+5, caption="")
        + batch_norm(f"{name}_bn", None, None, offset="(0,0,0)", origin=f"({name}_in-east)", width=1, height=height+5, depth=depth+5, caption="")
        + leaky_relu(f"{name}_lrelu",  None, None, offset="(0,0,0)", origin=f"({name}_bn-east)", width=1, height=height+5, depth=depth+5, caption="")
        + downsample(f"{name}_out", num_channels, size, offset="(0,0,0)", origin=f"({name}_lrelu-east)", width=width, height=height, depth=depth, caption=caption)
    )

def unpool_deconv_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", prev_width=1, width=1, height=40, depth=40, caption=" "):
    return (
        upsample(f"{name}_in",  None, None, offset, origin, prev_width, height, depth, caption="")
        + deconv2D(f"{name}_out", num_channels, size, offset="(0,0,0)", origin=f"({name}_in-east)", width=width, height=height, depth=depth, caption=caption)
    )

def unpool_deconv_batchnorm_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", prev_width=1, width=1, height=40, depth=40, caption=" "):
    return (
        upsample(f"{name}_in",  None, None, offset, origin, prev_width, height, depth, caption="")
        + deconv2D(f"{name}_deconv",  None, None, offset="(0,0,0)", origin=f"({name}_in-east)", width=width, height=height, depth=depth, caption="")
        + batch_norm(f"{name}_out", num_channels, size, offset="(0,0,0)", origin=f"({name}_deconv-east)", width=1, height=height, depth=depth, caption=caption)
    )

def unpool_deconv_tanh_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", prev_width=1, width=1, height=40, depth=40, caption=" "):
    return (
        upsample(f"{name}_in",  None, None, offset, origin, prev_width, height, depth, caption="")
        + deconv2D(f"{name}_deconv",  None, None, offset="(0,0,0)", origin=f"({name}_in-east)", width=width, height=height, depth=depth, caption="")
        + tanh(f"{name}_out", num_channels, size, offset="(0,0,0)", origin=f"({name}_deconv-east)", width=1, height=height, depth=depth, caption=caption)
    )

def strided_deconv_tanh_block(name, num_channels, size, offset="(0,0,0)", origin="(0,0,0)", prev_width=1, width=1, height=40, depth=40, caption=" "):
    return (
        deconv2D_strided(f"{name}_in", None, None, offset=offset, origin=origin, width=width, height=height, depth=depth, caption="")
        + tanh(f"{name}_out", num_channels, size, offset="(0,0,0)", origin=f"({name}_in-east)", width=1, height=height, depth=depth, caption=caption)
    )


def connection(start, end):
    return (
        r"""
        \draw [connection]  ("""+ start+ r"""-east)    -- node {\midarrow} ("""+ end + r"""-west);
        """
    )

# fmt: on


def end_document():
    return r"""
\end{tikzpicture}
\end{document}
"""


def generate_tex_document(arch, pathname="file.tex"):
    with open(pathname, "w") as f:
        for c in arch:
            # print(c)
            f.write(c)
