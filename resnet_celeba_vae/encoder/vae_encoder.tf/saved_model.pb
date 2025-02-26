�<
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��3
�
6residual_unit_5/batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86residual_unit_5/batch_normalization_14/moving_variance
�
Jresidual_unit_5/batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp6residual_unit_5/batch_normalization_14/moving_variance*
_output_shapes	
:�*
dtype0
�
2residual_unit_5/batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42residual_unit_5/batch_normalization_14/moving_mean
�
Fresidual_unit_5/batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp2residual_unit_5/batch_normalization_14/moving_mean*
_output_shapes	
:�*
dtype0
�
6residual_unit_5/batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86residual_unit_5/batch_normalization_13/moving_variance
�
Jresidual_unit_5/batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp6residual_unit_5/batch_normalization_13/moving_variance*
_output_shapes	
:�*
dtype0
�
2residual_unit_5/batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42residual_unit_5/batch_normalization_13/moving_mean
�
Fresidual_unit_5/batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp2residual_unit_5/batch_normalization_13/moving_mean*
_output_shapes	
:�*
dtype0
�
6residual_unit_5/batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86residual_unit_5/batch_normalization_12/moving_variance
�
Jresidual_unit_5/batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp6residual_unit_5/batch_normalization_12/moving_variance*
_output_shapes	
:�*
dtype0
�
2residual_unit_5/batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42residual_unit_5/batch_normalization_12/moving_mean
�
Fresidual_unit_5/batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp2residual_unit_5/batch_normalization_12/moving_mean*
_output_shapes	
:�*
dtype0
�
+residual_unit_5/batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_5/batch_normalization_14/beta
�
?residual_unit_5/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp+residual_unit_5/batch_normalization_14/beta*
_output_shapes	
:�*
dtype0
�
,residual_unit_5/batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,residual_unit_5/batch_normalization_14/gamma
�
@residual_unit_5/batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp,residual_unit_5/batch_normalization_14/gamma*
_output_shapes	
:�*
dtype0
�
 residual_unit_5/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" residual_unit_5/conv2d_15/kernel
�
4residual_unit_5/conv2d_15/kernel/Read/ReadVariableOpReadVariableOp residual_unit_5/conv2d_15/kernel*(
_output_shapes
:��*
dtype0
�
+residual_unit_5/batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_5/batch_normalization_13/beta
�
?residual_unit_5/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOp+residual_unit_5/batch_normalization_13/beta*
_output_shapes	
:�*
dtype0
�
,residual_unit_5/batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,residual_unit_5/batch_normalization_13/gamma
�
@residual_unit_5/batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOp,residual_unit_5/batch_normalization_13/gamma*
_output_shapes	
:�*
dtype0
�
 residual_unit_5/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" residual_unit_5/conv2d_14/kernel
�
4residual_unit_5/conv2d_14/kernel/Read/ReadVariableOpReadVariableOp residual_unit_5/conv2d_14/kernel*(
_output_shapes
:��*
dtype0
�
+residual_unit_5/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_5/batch_normalization_12/beta
�
?residual_unit_5/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp+residual_unit_5/batch_normalization_12/beta*
_output_shapes	
:�*
dtype0
�
,residual_unit_5/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,residual_unit_5/batch_normalization_12/gamma
�
@residual_unit_5/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp,residual_unit_5/batch_normalization_12/gamma*
_output_shapes	
:�*
dtype0
�
 residual_unit_5/conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" residual_unit_5/conv2d_13/kernel
�
4residual_unit_5/conv2d_13/kernel/Read/ReadVariableOpReadVariableOp residual_unit_5/conv2d_13/kernel*(
_output_shapes
:��*
dtype0
�
6residual_unit_4/batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86residual_unit_4/batch_normalization_11/moving_variance
�
Jresidual_unit_4/batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp6residual_unit_4/batch_normalization_11/moving_variance*
_output_shapes	
:�*
dtype0
�
2residual_unit_4/batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42residual_unit_4/batch_normalization_11/moving_mean
�
Fresidual_unit_4/batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp2residual_unit_4/batch_normalization_11/moving_mean*
_output_shapes	
:�*
dtype0
�
6residual_unit_4/batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86residual_unit_4/batch_normalization_10/moving_variance
�
Jresidual_unit_4/batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp6residual_unit_4/batch_normalization_10/moving_variance*
_output_shapes	
:�*
dtype0
�
2residual_unit_4/batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42residual_unit_4/batch_normalization_10/moving_mean
�
Fresidual_unit_4/batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp2residual_unit_4/batch_normalization_10/moving_mean*
_output_shapes	
:�*
dtype0
�
+residual_unit_4/batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_4/batch_normalization_11/beta
�
?residual_unit_4/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOp+residual_unit_4/batch_normalization_11/beta*
_output_shapes	
:�*
dtype0
�
,residual_unit_4/batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,residual_unit_4/batch_normalization_11/gamma
�
@residual_unit_4/batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOp,residual_unit_4/batch_normalization_11/gamma*
_output_shapes	
:�*
dtype0
�
 residual_unit_4/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" residual_unit_4/conv2d_12/kernel
�
4residual_unit_4/conv2d_12/kernel/Read/ReadVariableOpReadVariableOp residual_unit_4/conv2d_12/kernel*(
_output_shapes
:��*
dtype0
�
+residual_unit_4/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_4/batch_normalization_10/beta
�
?residual_unit_4/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp+residual_unit_4/batch_normalization_10/beta*
_output_shapes	
:�*
dtype0
�
,residual_unit_4/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,residual_unit_4/batch_normalization_10/gamma
�
@residual_unit_4/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp,residual_unit_4/batch_normalization_10/gamma*
_output_shapes	
:�*
dtype0
�
 residual_unit_4/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" residual_unit_4/conv2d_11/kernel
�
4residual_unit_4/conv2d_11/kernel/Read/ReadVariableOpReadVariableOp residual_unit_4/conv2d_11/kernel*(
_output_shapes
:��*
dtype0
�
5residual_unit_3/batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_3/batch_normalization_9/moving_variance
�
Iresidual_unit_3/batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_3/batch_normalization_9/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_3/batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_3/batch_normalization_9/moving_mean
�
Eresidual_unit_3/batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_3/batch_normalization_9/moving_mean*
_output_shapes	
:�*
dtype0
�
5residual_unit_3/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_3/batch_normalization_8/moving_variance
�
Iresidual_unit_3/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_3/batch_normalization_8/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_3/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_3/batch_normalization_8/moving_mean
�
Eresidual_unit_3/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_3/batch_normalization_8/moving_mean*
_output_shapes	
:�*
dtype0
�
5residual_unit_3/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_3/batch_normalization_7/moving_variance
�
Iresidual_unit_3/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_3/batch_normalization_7/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_3/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_3/batch_normalization_7/moving_mean
�
Eresidual_unit_3/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_3/batch_normalization_7/moving_mean*
_output_shapes	
:�*
dtype0
�
*residual_unit_3/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_3/batch_normalization_9/beta
�
>residual_unit_3/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp*residual_unit_3/batch_normalization_9/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_3/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_3/batch_normalization_9/gamma
�
?residual_unit_3/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_3/batch_normalization_9/gamma*
_output_shapes	
:�*
dtype0
�
 residual_unit_3/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*1
shared_name" residual_unit_3/conv2d_10/kernel
�
4residual_unit_3/conv2d_10/kernel/Read/ReadVariableOpReadVariableOp residual_unit_3/conv2d_10/kernel*(
_output_shapes
:��*
dtype0
�
*residual_unit_3/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_3/batch_normalization_8/beta
�
>residual_unit_3/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp*residual_unit_3/batch_normalization_8/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_3/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_3/batch_normalization_8/gamma
�
?residual_unit_3/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_3/batch_normalization_8/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_3/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!residual_unit_3/conv2d_9/kernel
�
3residual_unit_3/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_3/conv2d_9/kernel*(
_output_shapes
:��*
dtype0
�
*residual_unit_3/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_3/batch_normalization_7/beta
�
>residual_unit_3/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp*residual_unit_3/batch_normalization_7/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_3/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_3/batch_normalization_7/gamma
�
?residual_unit_3/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_3/batch_normalization_7/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_3/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!residual_unit_3/conv2d_8/kernel
�
3residual_unit_3/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_3/conv2d_8/kernel*(
_output_shapes
:��*
dtype0
�
5residual_unit_2/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_2/batch_normalization_6/moving_variance
�
Iresidual_unit_2/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_2/batch_normalization_6/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_2/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_2/batch_normalization_6/moving_mean
�
Eresidual_unit_2/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_2/batch_normalization_6/moving_mean*
_output_shapes	
:�*
dtype0
�
5residual_unit_2/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_2/batch_normalization_5/moving_variance
�
Iresidual_unit_2/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_2/batch_normalization_5/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_2/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_2/batch_normalization_5/moving_mean
�
Eresidual_unit_2/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_2/batch_normalization_5/moving_mean*
_output_shapes	
:�*
dtype0
�
*residual_unit_2/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_2/batch_normalization_6/beta
�
>residual_unit_2/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp*residual_unit_2/batch_normalization_6/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_2/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_2/batch_normalization_6/gamma
�
?residual_unit_2/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_2/batch_normalization_6/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_2/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!residual_unit_2/conv2d_7/kernel
�
3residual_unit_2/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_2/conv2d_7/kernel*(
_output_shapes
:��*
dtype0
�
*residual_unit_2/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_2/batch_normalization_5/beta
�
>residual_unit_2/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp*residual_unit_2/batch_normalization_5/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_2/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_2/batch_normalization_5/gamma
�
?residual_unit_2/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_2/batch_normalization_5/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_2/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!residual_unit_2/conv2d_6/kernel
�
3residual_unit_2/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_2/conv2d_6/kernel*(
_output_shapes
:��*
dtype0
�
5residual_unit_1/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_1/batch_normalization_4/moving_variance
�
Iresidual_unit_1/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_1/batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_1/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_1/batch_normalization_4/moving_mean
�
Eresidual_unit_1/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_1/batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
5residual_unit_1/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_1/batch_normalization_3/moving_variance
�
Iresidual_unit_1/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_1/batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_1/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_1/batch_normalization_3/moving_mean
�
Eresidual_unit_1/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_1/batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
5residual_unit_1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75residual_unit_1/batch_normalization_2/moving_variance
�
Iresidual_unit_1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp5residual_unit_1/batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
1residual_unit_1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31residual_unit_1/batch_normalization_2/moving_mean
�
Eresidual_unit_1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp1residual_unit_1/batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
*residual_unit_1/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_1/batch_normalization_4/beta
�
>residual_unit_1/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp*residual_unit_1/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_1/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_1/batch_normalization_4/gamma
�
?residual_unit_1/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_1/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*0
shared_name!residual_unit_1/conv2d_5/kernel
�
3residual_unit_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_1/conv2d_5/kernel*'
_output_shapes
:@�*
dtype0
�
*residual_unit_1/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_1/batch_normalization_3/beta
�
>residual_unit_1/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp*residual_unit_1/batch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_1/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_1/batch_normalization_3/gamma
�
?residual_unit_1/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_1/batch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_1/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!residual_unit_1/conv2d_4/kernel
�
3residual_unit_1/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_1/conv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
*residual_unit_1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*residual_unit_1/batch_normalization_2/beta
�
>residual_unit_1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp*residual_unit_1/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
+residual_unit_1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*<
shared_name-+residual_unit_1/batch_normalization_2/gamma
�
?residual_unit_1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp+residual_unit_1/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
residual_unit_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*0
shared_name!residual_unit_1/conv2d_3/kernel
�
3residual_unit_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpresidual_unit_1/conv2d_3/kernel*'
_output_shapes
:@�*
dtype0
�
3residual_unit/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53residual_unit/batch_normalization_1/moving_variance
�
Gresidual_unit/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp3residual_unit/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
/residual_unit/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/residual_unit/batch_normalization_1/moving_mean
�
Cresidual_unit/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp/residual_unit/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
1residual_unit/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31residual_unit/batch_normalization/moving_variance
�
Eresidual_unit/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp1residual_unit/batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
�
-residual_unit/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-residual_unit/batch_normalization/moving_mean
�
Aresidual_unit/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp-residual_unit/batch_normalization/moving_mean*
_output_shapes
:@*
dtype0
�
(residual_unit/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(residual_unit/batch_normalization_1/beta
�
<residual_unit/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp(residual_unit/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
)residual_unit/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)residual_unit/batch_normalization_1/gamma
�
=residual_unit/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp)residual_unit/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
residual_unit/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresidual_unit/conv2d_2/kernel
�
1residual_unit/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpresidual_unit/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
�
&residual_unit/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&residual_unit/batch_normalization/beta
�
:residual_unit/batch_normalization/beta/Read/ReadVariableOpReadVariableOp&residual_unit/batch_normalization/beta*
_output_shapes
:@*
dtype0
�
'residual_unit/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'residual_unit/batch_normalization/gamma
�
;residual_unit/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp'residual_unit/batch_normalization/gamma*
_output_shapes
:@*
dtype0
�
residual_unit/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameresidual_unit/conv2d_1/kernel
�
1residual_unit/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpresidual_unit/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelresidual_unit/conv2d_1/kernel'residual_unit/batch_normalization/gamma&residual_unit/batch_normalization/beta-residual_unit/batch_normalization/moving_mean1residual_unit/batch_normalization/moving_varianceresidual_unit/conv2d_2/kernel)residual_unit/batch_normalization_1/gamma(residual_unit/batch_normalization_1/beta/residual_unit/batch_normalization_1/moving_mean3residual_unit/batch_normalization_1/moving_varianceresidual_unit_1/conv2d_3/kernel+residual_unit_1/batch_normalization_2/gamma*residual_unit_1/batch_normalization_2/beta1residual_unit_1/batch_normalization_2/moving_mean5residual_unit_1/batch_normalization_2/moving_varianceresidual_unit_1/conv2d_4/kernel+residual_unit_1/batch_normalization_3/gamma*residual_unit_1/batch_normalization_3/beta1residual_unit_1/batch_normalization_3/moving_mean5residual_unit_1/batch_normalization_3/moving_varianceresidual_unit_1/conv2d_5/kernel+residual_unit_1/batch_normalization_4/gamma*residual_unit_1/batch_normalization_4/beta1residual_unit_1/batch_normalization_4/moving_mean5residual_unit_1/batch_normalization_4/moving_varianceresidual_unit_2/conv2d_6/kernel+residual_unit_2/batch_normalization_5/gamma*residual_unit_2/batch_normalization_5/beta1residual_unit_2/batch_normalization_5/moving_mean5residual_unit_2/batch_normalization_5/moving_varianceresidual_unit_2/conv2d_7/kernel+residual_unit_2/batch_normalization_6/gamma*residual_unit_2/batch_normalization_6/beta1residual_unit_2/batch_normalization_6/moving_mean5residual_unit_2/batch_normalization_6/moving_varianceresidual_unit_3/conv2d_8/kernel+residual_unit_3/batch_normalization_7/gamma*residual_unit_3/batch_normalization_7/beta1residual_unit_3/batch_normalization_7/moving_mean5residual_unit_3/batch_normalization_7/moving_varianceresidual_unit_3/conv2d_9/kernel+residual_unit_3/batch_normalization_8/gamma*residual_unit_3/batch_normalization_8/beta1residual_unit_3/batch_normalization_8/moving_mean5residual_unit_3/batch_normalization_8/moving_variance residual_unit_3/conv2d_10/kernel+residual_unit_3/batch_normalization_9/gamma*residual_unit_3/batch_normalization_9/beta1residual_unit_3/batch_normalization_9/moving_mean5residual_unit_3/batch_normalization_9/moving_variance residual_unit_4/conv2d_11/kernel,residual_unit_4/batch_normalization_10/gamma+residual_unit_4/batch_normalization_10/beta2residual_unit_4/batch_normalization_10/moving_mean6residual_unit_4/batch_normalization_10/moving_variance residual_unit_4/conv2d_12/kernel,residual_unit_4/batch_normalization_11/gamma+residual_unit_4/batch_normalization_11/beta2residual_unit_4/batch_normalization_11/moving_mean6residual_unit_4/batch_normalization_11/moving_variance residual_unit_5/conv2d_13/kernel,residual_unit_5/batch_normalization_12/gamma+residual_unit_5/batch_normalization_12/beta2residual_unit_5/batch_normalization_12/moving_mean6residual_unit_5/batch_normalization_12/moving_variance residual_unit_5/conv2d_14/kernel,residual_unit_5/batch_normalization_13/gamma+residual_unit_5/batch_normalization_13/beta2residual_unit_5/batch_normalization_13/moving_mean6residual_unit_5/batch_normalization_13/moving_variance residual_unit_5/conv2d_15/kernel,residual_unit_5/batch_normalization_14/gamma+residual_unit_5/batch_normalization_14/beta2residual_unit_5/batch_normalization_14/moving_mean6residual_unit_5/batch_normalization_14/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_128020684

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�Bܶ BԶ
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

_init_input_shape* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
  _jit_compiled_convolution_op*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses* 
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3main_layers
4skip_layers*
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;main_layers
<skip_layers*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cmain_layers
Dskip_layers*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kmain_layers
Lskip_layers*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Smain_layers
Tskip_layers*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[main_layers
\skip_layers*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses* 
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias*
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 
�
0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
o76
p77
w78
x79*
�
0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
o46
p47
w48
x49*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 
* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
S
0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
3
0
�1
�2
�3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
$
�0
�1
�3
�4*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
L
�0
�1
�2
�3
�4
�5
�6
�7
�8*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
$
�0
�1
�3
�4*

�0
�1*
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
4
�0
�1
�2
�3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
$
�0
�1
�3
�4*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
L
�0
�1
�2
�3
�4
�5
�6
�7
�8*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
$
�0
�1
�3
�4*

�0
�1*
T
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9*
4
�0
�1
�2
�3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
$
�0
�1
�3
�4*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
L
�0
�1
�2
�3
�4
�5
�6
�7
�8*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
$
�0
�1
�3
�4*

�0
�1*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

o0
p1*

o0
p1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEresidual_unit/conv2d_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'residual_unit/batch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&residual_unit/batch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEresidual_unit/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)residual_unit/batch_normalization_1/gamma&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(residual_unit/batch_normalization_1/beta&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-residual_unit/batch_normalization/moving_mean&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1residual_unit/batch_normalization/moving_variance&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/residual_unit/batch_normalization_1/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3residual_unit/batch_normalization_1/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_1/conv2d_3/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_1/batch_normalization_2/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_1/batch_normalization_2/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_1/conv2d_4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_1/batch_normalization_3/gamma'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_1/batch_normalization_3/beta'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_1/conv2d_5/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_1/batch_normalization_4/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_1/batch_normalization_4/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_1/batch_normalization_2/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_1/batch_normalization_2/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_1/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_1/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_1/batch_normalization_4/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_1/batch_normalization_4/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_2/conv2d_6/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_2/batch_normalization_5/gamma'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_2/batch_normalization_5/beta'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_2/conv2d_7/kernel'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_2/batch_normalization_6/gamma'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_2/batch_normalization_6/beta'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_2/batch_normalization_5/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_2/batch_normalization_5/moving_variance'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_2/batch_normalization_6/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_2/batch_normalization_6/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_3/conv2d_8/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_3/batch_normalization_7/gamma'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_3/batch_normalization_7/beta'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEresidual_unit_3/conv2d_9/kernel'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_3/batch_normalization_8/gamma'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_3/batch_normalization_8/beta'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE residual_unit_3/conv2d_10/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_3/batch_normalization_9/gamma'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*residual_unit_3/batch_normalization_9/beta'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_3/batch_normalization_7/moving_mean'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_3/batch_normalization_7/moving_variance'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_3/batch_normalization_8/moving_mean'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_3/batch_normalization_8/moving_variance'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1residual_unit_3/batch_normalization_9/moving_mean'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5residual_unit_3/batch_normalization_9/moving_variance'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE residual_unit_4/conv2d_11/kernel'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,residual_unit_4/batch_normalization_10/gamma'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_4/batch_normalization_10/beta'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE residual_unit_4/conv2d_12/kernel'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,residual_unit_4/batch_normalization_11/gamma'variables/55/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_4/batch_normalization_11/beta'variables/56/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2residual_unit_4/batch_normalization_10/moving_mean'variables/57/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6residual_unit_4/batch_normalization_10/moving_variance'variables/58/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2residual_unit_4/batch_normalization_11/moving_mean'variables/59/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6residual_unit_4/batch_normalization_11/moving_variance'variables/60/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE residual_unit_5/conv2d_13/kernel'variables/61/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,residual_unit_5/batch_normalization_12/gamma'variables/62/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_5/batch_normalization_12/beta'variables/63/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE residual_unit_5/conv2d_14/kernel'variables/64/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,residual_unit_5/batch_normalization_13/gamma'variables/65/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_5/batch_normalization_13/beta'variables/66/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE residual_unit_5/conv2d_15/kernel'variables/67/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,residual_unit_5/batch_normalization_14/gamma'variables/68/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+residual_unit_5/batch_normalization_14/beta'variables/69/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2residual_unit_5/batch_normalization_12/moving_mean'variables/70/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6residual_unit_5/batch_normalization_12/moving_variance'variables/71/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2residual_unit_5/batch_normalization_13/moving_mean'variables/72/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6residual_unit_5/batch_normalization_13/moving_variance'variables/73/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2residual_unit_5/batch_normalization_14/moving_mean'variables/74/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6residual_unit_5/batch_normalization_14/moving_variance'variables/75/.ATTRIBUTES/VARIABLE_VALUE*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29*
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 
* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp1residual_unit/conv2d_1/kernel/Read/ReadVariableOp;residual_unit/batch_normalization/gamma/Read/ReadVariableOp:residual_unit/batch_normalization/beta/Read/ReadVariableOp1residual_unit/conv2d_2/kernel/Read/ReadVariableOp=residual_unit/batch_normalization_1/gamma/Read/ReadVariableOp<residual_unit/batch_normalization_1/beta/Read/ReadVariableOpAresidual_unit/batch_normalization/moving_mean/Read/ReadVariableOpEresidual_unit/batch_normalization/moving_variance/Read/ReadVariableOpCresidual_unit/batch_normalization_1/moving_mean/Read/ReadVariableOpGresidual_unit/batch_normalization_1/moving_variance/Read/ReadVariableOp3residual_unit_1/conv2d_3/kernel/Read/ReadVariableOp?residual_unit_1/batch_normalization_2/gamma/Read/ReadVariableOp>residual_unit_1/batch_normalization_2/beta/Read/ReadVariableOp3residual_unit_1/conv2d_4/kernel/Read/ReadVariableOp?residual_unit_1/batch_normalization_3/gamma/Read/ReadVariableOp>residual_unit_1/batch_normalization_3/beta/Read/ReadVariableOp3residual_unit_1/conv2d_5/kernel/Read/ReadVariableOp?residual_unit_1/batch_normalization_4/gamma/Read/ReadVariableOp>residual_unit_1/batch_normalization_4/beta/Read/ReadVariableOpEresidual_unit_1/batch_normalization_2/moving_mean/Read/ReadVariableOpIresidual_unit_1/batch_normalization_2/moving_variance/Read/ReadVariableOpEresidual_unit_1/batch_normalization_3/moving_mean/Read/ReadVariableOpIresidual_unit_1/batch_normalization_3/moving_variance/Read/ReadVariableOpEresidual_unit_1/batch_normalization_4/moving_mean/Read/ReadVariableOpIresidual_unit_1/batch_normalization_4/moving_variance/Read/ReadVariableOp3residual_unit_2/conv2d_6/kernel/Read/ReadVariableOp?residual_unit_2/batch_normalization_5/gamma/Read/ReadVariableOp>residual_unit_2/batch_normalization_5/beta/Read/ReadVariableOp3residual_unit_2/conv2d_7/kernel/Read/ReadVariableOp?residual_unit_2/batch_normalization_6/gamma/Read/ReadVariableOp>residual_unit_2/batch_normalization_6/beta/Read/ReadVariableOpEresidual_unit_2/batch_normalization_5/moving_mean/Read/ReadVariableOpIresidual_unit_2/batch_normalization_5/moving_variance/Read/ReadVariableOpEresidual_unit_2/batch_normalization_6/moving_mean/Read/ReadVariableOpIresidual_unit_2/batch_normalization_6/moving_variance/Read/ReadVariableOp3residual_unit_3/conv2d_8/kernel/Read/ReadVariableOp?residual_unit_3/batch_normalization_7/gamma/Read/ReadVariableOp>residual_unit_3/batch_normalization_7/beta/Read/ReadVariableOp3residual_unit_3/conv2d_9/kernel/Read/ReadVariableOp?residual_unit_3/batch_normalization_8/gamma/Read/ReadVariableOp>residual_unit_3/batch_normalization_8/beta/Read/ReadVariableOp4residual_unit_3/conv2d_10/kernel/Read/ReadVariableOp?residual_unit_3/batch_normalization_9/gamma/Read/ReadVariableOp>residual_unit_3/batch_normalization_9/beta/Read/ReadVariableOpEresidual_unit_3/batch_normalization_7/moving_mean/Read/ReadVariableOpIresidual_unit_3/batch_normalization_7/moving_variance/Read/ReadVariableOpEresidual_unit_3/batch_normalization_8/moving_mean/Read/ReadVariableOpIresidual_unit_3/batch_normalization_8/moving_variance/Read/ReadVariableOpEresidual_unit_3/batch_normalization_9/moving_mean/Read/ReadVariableOpIresidual_unit_3/batch_normalization_9/moving_variance/Read/ReadVariableOp4residual_unit_4/conv2d_11/kernel/Read/ReadVariableOp@residual_unit_4/batch_normalization_10/gamma/Read/ReadVariableOp?residual_unit_4/batch_normalization_10/beta/Read/ReadVariableOp4residual_unit_4/conv2d_12/kernel/Read/ReadVariableOp@residual_unit_4/batch_normalization_11/gamma/Read/ReadVariableOp?residual_unit_4/batch_normalization_11/beta/Read/ReadVariableOpFresidual_unit_4/batch_normalization_10/moving_mean/Read/ReadVariableOpJresidual_unit_4/batch_normalization_10/moving_variance/Read/ReadVariableOpFresidual_unit_4/batch_normalization_11/moving_mean/Read/ReadVariableOpJresidual_unit_4/batch_normalization_11/moving_variance/Read/ReadVariableOp4residual_unit_5/conv2d_13/kernel/Read/ReadVariableOp@residual_unit_5/batch_normalization_12/gamma/Read/ReadVariableOp?residual_unit_5/batch_normalization_12/beta/Read/ReadVariableOp4residual_unit_5/conv2d_14/kernel/Read/ReadVariableOp@residual_unit_5/batch_normalization_13/gamma/Read/ReadVariableOp?residual_unit_5/batch_normalization_13/beta/Read/ReadVariableOp4residual_unit_5/conv2d_15/kernel/Read/ReadVariableOp@residual_unit_5/batch_normalization_14/gamma/Read/ReadVariableOp?residual_unit_5/batch_normalization_14/beta/Read/ReadVariableOpFresidual_unit_5/batch_normalization_12/moving_mean/Read/ReadVariableOpJresidual_unit_5/batch_normalization_12/moving_variance/Read/ReadVariableOpFresidual_unit_5/batch_normalization_13/moving_mean/Read/ReadVariableOpJresidual_unit_5/batch_normalization_13/moving_variance/Read/ReadVariableOpFresidual_unit_5/batch_normalization_14/moving_mean/Read/ReadVariableOpJresidual_unit_5/batch_normalization_14/moving_variance/Read/ReadVariableOpConst*]
TinV
T2R*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_save_128023909
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasresidual_unit/conv2d_1/kernel'residual_unit/batch_normalization/gamma&residual_unit/batch_normalization/betaresidual_unit/conv2d_2/kernel)residual_unit/batch_normalization_1/gamma(residual_unit/batch_normalization_1/beta-residual_unit/batch_normalization/moving_mean1residual_unit/batch_normalization/moving_variance/residual_unit/batch_normalization_1/moving_mean3residual_unit/batch_normalization_1/moving_varianceresidual_unit_1/conv2d_3/kernel+residual_unit_1/batch_normalization_2/gamma*residual_unit_1/batch_normalization_2/betaresidual_unit_1/conv2d_4/kernel+residual_unit_1/batch_normalization_3/gamma*residual_unit_1/batch_normalization_3/betaresidual_unit_1/conv2d_5/kernel+residual_unit_1/batch_normalization_4/gamma*residual_unit_1/batch_normalization_4/beta1residual_unit_1/batch_normalization_2/moving_mean5residual_unit_1/batch_normalization_2/moving_variance1residual_unit_1/batch_normalization_3/moving_mean5residual_unit_1/batch_normalization_3/moving_variance1residual_unit_1/batch_normalization_4/moving_mean5residual_unit_1/batch_normalization_4/moving_varianceresidual_unit_2/conv2d_6/kernel+residual_unit_2/batch_normalization_5/gamma*residual_unit_2/batch_normalization_5/betaresidual_unit_2/conv2d_7/kernel+residual_unit_2/batch_normalization_6/gamma*residual_unit_2/batch_normalization_6/beta1residual_unit_2/batch_normalization_5/moving_mean5residual_unit_2/batch_normalization_5/moving_variance1residual_unit_2/batch_normalization_6/moving_mean5residual_unit_2/batch_normalization_6/moving_varianceresidual_unit_3/conv2d_8/kernel+residual_unit_3/batch_normalization_7/gamma*residual_unit_3/batch_normalization_7/betaresidual_unit_3/conv2d_9/kernel+residual_unit_3/batch_normalization_8/gamma*residual_unit_3/batch_normalization_8/beta residual_unit_3/conv2d_10/kernel+residual_unit_3/batch_normalization_9/gamma*residual_unit_3/batch_normalization_9/beta1residual_unit_3/batch_normalization_7/moving_mean5residual_unit_3/batch_normalization_7/moving_variance1residual_unit_3/batch_normalization_8/moving_mean5residual_unit_3/batch_normalization_8/moving_variance1residual_unit_3/batch_normalization_9/moving_mean5residual_unit_3/batch_normalization_9/moving_variance residual_unit_4/conv2d_11/kernel,residual_unit_4/batch_normalization_10/gamma+residual_unit_4/batch_normalization_10/beta residual_unit_4/conv2d_12/kernel,residual_unit_4/batch_normalization_11/gamma+residual_unit_4/batch_normalization_11/beta2residual_unit_4/batch_normalization_10/moving_mean6residual_unit_4/batch_normalization_10/moving_variance2residual_unit_4/batch_normalization_11/moving_mean6residual_unit_4/batch_normalization_11/moving_variance residual_unit_5/conv2d_13/kernel,residual_unit_5/batch_normalization_12/gamma+residual_unit_5/batch_normalization_12/beta residual_unit_5/conv2d_14/kernel,residual_unit_5/batch_normalization_13/gamma+residual_unit_5/batch_normalization_13/beta residual_unit_5/conv2d_15/kernel,residual_unit_5/batch_normalization_14/gamma+residual_unit_5/batch_normalization_14/beta2residual_unit_5/batch_normalization_12/moving_mean6residual_unit_5/batch_normalization_12/moving_variance2residual_unit_5/batch_normalization_13/moving_mean6residual_unit_5/batch_normalization_13/moving_variance2residual_unit_5/batch_normalization_14/moving_mean6residual_unit_5/batch_normalization_14/moving_variance*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference__traced_restore_128024159��.
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128017795

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�/
�	
L__inference_residual_unit_layer_call_and_return_conditional_losses_128018109

inputsA
'conv2d_1_conv2d_readvariableop_resource:@@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( p
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( z
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:���������  @Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:���������  @k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:���������  @�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  @: : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�

�
3__inference_residual_unit_2_layer_call_fn_128022021

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128018262x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128017826

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
3__inference_residual_unit_4_layer_call_fn_128022364

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018987x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
1__inference_residual_unit_layer_call_fn_128021728

inputs!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@#
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_residual_unit_layer_call_and_return_conditional_losses_128019429w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  @: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
��
�h
D__inference_model_layer_call_and_return_conditional_losses_128021644

inputs?
%conv2d_conv2d_readvariableop_resource:@O
5residual_unit_conv2d_1_conv2d_readvariableop_resource:@@G
9residual_unit_batch_normalization_readvariableop_resource:@I
;residual_unit_batch_normalization_readvariableop_1_resource:@X
Jresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource:@Z
Lresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@O
5residual_unit_conv2d_2_conv2d_readvariableop_resource:@@I
;residual_unit_batch_normalization_1_readvariableop_resource:@K
=residual_unit_batch_normalization_1_readvariableop_1_resource:@Z
Lresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@\
Nresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@R
7residual_unit_1_conv2d_3_conv2d_readvariableop_resource:@�L
=residual_unit_1_batch_normalization_2_readvariableop_resource:	�N
?residual_unit_1_batch_normalization_2_readvariableop_1_resource:	�]
Nresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_1_conv2d_4_conv2d_readvariableop_resource:��L
=residual_unit_1_batch_normalization_3_readvariableop_resource:	�N
?residual_unit_1_batch_normalization_3_readvariableop_1_resource:	�]
Nresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�R
7residual_unit_1_conv2d_5_conv2d_readvariableop_resource:@�L
=residual_unit_1_batch_normalization_4_readvariableop_resource:	�N
?residual_unit_1_batch_normalization_4_readvariableop_1_resource:	�]
Nresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_2_conv2d_6_conv2d_readvariableop_resource:��L
=residual_unit_2_batch_normalization_5_readvariableop_resource:	�N
?residual_unit_2_batch_normalization_5_readvariableop_1_resource:	�]
Nresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_2_conv2d_7_conv2d_readvariableop_resource:��L
=residual_unit_2_batch_normalization_6_readvariableop_resource:	�N
?residual_unit_2_batch_normalization_6_readvariableop_1_resource:	�]
Nresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_3_conv2d_8_conv2d_readvariableop_resource:��L
=residual_unit_3_batch_normalization_7_readvariableop_resource:	�N
?residual_unit_3_batch_normalization_7_readvariableop_1_resource:	�]
Nresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_3_conv2d_9_conv2d_readvariableop_resource:��L
=residual_unit_3_batch_normalization_8_readvariableop_resource:	�N
?residual_unit_3_batch_normalization_8_readvariableop_1_resource:	�]
Nresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_3_conv2d_10_conv2d_readvariableop_resource:��L
=residual_unit_3_batch_normalization_9_readvariableop_resource:	�N
?residual_unit_3_batch_normalization_9_readvariableop_1_resource:	�]
Nresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_4_conv2d_11_conv2d_readvariableop_resource:��M
>residual_unit_4_batch_normalization_10_readvariableop_resource:	�O
@residual_unit_4_batch_normalization_10_readvariableop_1_resource:	�^
Oresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_4_conv2d_12_conv2d_readvariableop_resource:��M
>residual_unit_4_batch_normalization_11_readvariableop_resource:	�O
@residual_unit_4_batch_normalization_11_readvariableop_1_resource:	�^
Oresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_5_conv2d_13_conv2d_readvariableop_resource:��M
>residual_unit_5_batch_normalization_12_readvariableop_resource:	�O
@residual_unit_5_batch_normalization_12_readvariableop_1_resource:	�^
Oresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_5_conv2d_14_conv2d_readvariableop_resource:��M
>residual_unit_5_batch_normalization_13_readvariableop_resource:	�O
@residual_unit_5_batch_normalization_13_readvariableop_1_resource:	�^
Oresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_5_conv2d_15_conv2d_readvariableop_resource:��M
>residual_unit_5_batch_normalization_14_readvariableop_resource:	�O
@residual_unit_5_batch_normalization_14_readvariableop_1_resource:	�^
Oresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��conv2d/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0residual_unit/batch_normalization/AssignNewValue�2residual_unit/batch_normalization/AssignNewValue_1�Aresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp�Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�0residual_unit/batch_normalization/ReadVariableOp�2residual_unit/batch_normalization/ReadVariableOp_1�2residual_unit/batch_normalization_1/AssignNewValue�4residual_unit/batch_normalization_1/AssignNewValue_1�Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�2residual_unit/batch_normalization_1/ReadVariableOp�4residual_unit/batch_normalization_1/ReadVariableOp_1�,residual_unit/conv2d_1/Conv2D/ReadVariableOp�,residual_unit/conv2d_2/Conv2D/ReadVariableOp�4residual_unit_1/batch_normalization_2/AssignNewValue�6residual_unit_1/batch_normalization_2/AssignNewValue_1�Eresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_1/batch_normalization_2/ReadVariableOp�6residual_unit_1/batch_normalization_2/ReadVariableOp_1�4residual_unit_1/batch_normalization_3/AssignNewValue�6residual_unit_1/batch_normalization_3/AssignNewValue_1�Eresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_1/batch_normalization_3/ReadVariableOp�6residual_unit_1/batch_normalization_3/ReadVariableOp_1�4residual_unit_1/batch_normalization_4/AssignNewValue�6residual_unit_1/batch_normalization_4/AssignNewValue_1�Eresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_1/batch_normalization_4/ReadVariableOp�6residual_unit_1/batch_normalization_4/ReadVariableOp_1�.residual_unit_1/conv2d_3/Conv2D/ReadVariableOp�.residual_unit_1/conv2d_4/Conv2D/ReadVariableOp�.residual_unit_1/conv2d_5/Conv2D/ReadVariableOp�4residual_unit_2/batch_normalization_5/AssignNewValue�6residual_unit_2/batch_normalization_5/AssignNewValue_1�Eresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_2/batch_normalization_5/ReadVariableOp�6residual_unit_2/batch_normalization_5/ReadVariableOp_1�4residual_unit_2/batch_normalization_6/AssignNewValue�6residual_unit_2/batch_normalization_6/AssignNewValue_1�Eresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_2/batch_normalization_6/ReadVariableOp�6residual_unit_2/batch_normalization_6/ReadVariableOp_1�.residual_unit_2/conv2d_6/Conv2D/ReadVariableOp�.residual_unit_2/conv2d_7/Conv2D/ReadVariableOp�4residual_unit_3/batch_normalization_7/AssignNewValue�6residual_unit_3/batch_normalization_7/AssignNewValue_1�Eresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_3/batch_normalization_7/ReadVariableOp�6residual_unit_3/batch_normalization_7/ReadVariableOp_1�4residual_unit_3/batch_normalization_8/AssignNewValue�6residual_unit_3/batch_normalization_8/AssignNewValue_1�Eresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_3/batch_normalization_8/ReadVariableOp�6residual_unit_3/batch_normalization_8/ReadVariableOp_1�4residual_unit_3/batch_normalization_9/AssignNewValue�6residual_unit_3/batch_normalization_9/AssignNewValue_1�Eresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_3/batch_normalization_9/ReadVariableOp�6residual_unit_3/batch_normalization_9/ReadVariableOp_1�/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp�.residual_unit_3/conv2d_8/Conv2D/ReadVariableOp�.residual_unit_3/conv2d_9/Conv2D/ReadVariableOp�5residual_unit_4/batch_normalization_10/AssignNewValue�7residual_unit_4/batch_normalization_10/AssignNewValue_1�Fresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_4/batch_normalization_10/ReadVariableOp�7residual_unit_4/batch_normalization_10/ReadVariableOp_1�5residual_unit_4/batch_normalization_11/AssignNewValue�7residual_unit_4/batch_normalization_11/AssignNewValue_1�Fresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_4/batch_normalization_11/ReadVariableOp�7residual_unit_4/batch_normalization_11/ReadVariableOp_1�/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp�/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp�5residual_unit_5/batch_normalization_12/AssignNewValue�7residual_unit_5/batch_normalization_12/AssignNewValue_1�Fresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_5/batch_normalization_12/ReadVariableOp�7residual_unit_5/batch_normalization_12/ReadVariableOp_1�5residual_unit_5/batch_normalization_13/AssignNewValue�7residual_unit_5/batch_normalization_13/AssignNewValue_1�Fresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_5/batch_normalization_13/ReadVariableOp�7residual_unit_5/batch_normalization_13/ReadVariableOp_1�5residual_unit_5/batch_normalization_14/AssignNewValue�7residual_unit_5/batch_normalization_14/AssignNewValue_1�Fresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_5/batch_normalization_14/ReadVariableOp�7residual_unit_5/batch_normalization_14/ReadVariableOp_1�/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp�/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp�/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
i
activation/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������@@@�
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingSAME*
strides
�
,residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
residual_unit/conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:04residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
0residual_unit/batch_normalization/ReadVariableOpReadVariableOp9residual_unit_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
2residual_unit/batch_normalization/ReadVariableOp_1ReadVariableOp;residual_unit_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Aresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpJresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
2residual_unit/batch_normalization/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_1/Conv2D:output:08residual_unit/batch_normalization/ReadVariableOp:value:0:residual_unit/batch_normalization/ReadVariableOp_1:value:0Iresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Kresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
0residual_unit/batch_normalization/AssignNewValueAssignVariableOpJresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource?residual_unit/batch_normalization/FusedBatchNormV3:batch_mean:0B^residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
2residual_unit/batch_normalization/AssignNewValue_1AssignVariableOpLresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceCresidual_unit/batch_normalization/FusedBatchNormV3:batch_variance:0D^residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit/ReluRelu6residual_unit/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
,residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
residual_unit/conv2d_2/Conv2DConv2D residual_unit/Relu:activations:04residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
2residual_unit/batch_normalization_1/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
4residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
4residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_2/Conv2D:output:0:residual_unit/batch_normalization_1/ReadVariableOp:value:0<residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
2residual_unit/batch_normalization_1/AssignNewValueAssignVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceAresidual_unit/batch_normalization_1/FusedBatchNormV3:batch_mean:0D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
4residual_unit/batch_normalization_1/AssignNewValue_1AssignVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceEresidual_unit/batch_normalization_1/FusedBatchNormV3:batch_variance:0F^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit/addAddV28residual_unit/batch_normalization_1/FusedBatchNormV3:y:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:���������  @m
residual_unit/Relu_1Reluresidual_unit/add:z:0*
T0*/
_output_shapes
:���������  @�
.residual_unit_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp7residual_unit_1_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
residual_unit_1/conv2d_3/Conv2DConv2D"residual_unit/Relu_1:activations:06residual_unit_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_1/batch_normalization_2/ReadVariableOpReadVariableOp=residual_unit_1_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp?residual_unit_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3(residual_unit_1/conv2d_3/Conv2D:output:0<residual_unit_1/batch_normalization_2/ReadVariableOp:value:0>residual_unit_1/batch_normalization_2/ReadVariableOp_1:value:0Mresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_1/batch_normalization_2/AssignNewValueAssignVariableOpNresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceCresidual_unit_1/batch_normalization_2/FusedBatchNormV3:batch_mean:0F^residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_1/batch_normalization_2/AssignNewValue_1AssignVariableOpPresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_1/batch_normalization_2/FusedBatchNormV3:batch_variance:0H^residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_1/ReluRelu:residual_unit_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
.residual_unit_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp7residual_unit_1_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_1/conv2d_4/Conv2DConv2D"residual_unit_1/Relu:activations:06residual_unit_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_1/batch_normalization_3/ReadVariableOpReadVariableOp=residual_unit_1_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp?residual_unit_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3(residual_unit_1/conv2d_4/Conv2D:output:0<residual_unit_1/batch_normalization_3/ReadVariableOp:value:0>residual_unit_1/batch_normalization_3/ReadVariableOp_1:value:0Mresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_1/batch_normalization_3/AssignNewValueAssignVariableOpNresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceCresidual_unit_1/batch_normalization_3/FusedBatchNormV3:batch_mean:0F^residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_1/batch_normalization_3/AssignNewValue_1AssignVariableOpPresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_1/batch_normalization_3/FusedBatchNormV3:batch_variance:0H^residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
.residual_unit_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7residual_unit_1_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
residual_unit_1/conv2d_5/Conv2DConv2D"residual_unit/Relu_1:activations:06residual_unit_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_1/batch_normalization_4/ReadVariableOpReadVariableOp=residual_unit_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp?residual_unit_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3(residual_unit_1/conv2d_5/Conv2D:output:0<residual_unit_1/batch_normalization_4/ReadVariableOp:value:0>residual_unit_1/batch_normalization_4/ReadVariableOp_1:value:0Mresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_1/batch_normalization_4/AssignNewValueAssignVariableOpNresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceCresidual_unit_1/batch_normalization_4/FusedBatchNormV3:batch_mean:0F^residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_1/batch_normalization_4/AssignNewValue_1AssignVariableOpPresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_1/batch_normalization_4/FusedBatchNormV3:batch_variance:0H^residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_1/addAddV2:residual_unit_1/batch_normalization_3/FusedBatchNormV3:y:0:residual_unit_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������r
residual_unit_1/Relu_1Reluresidual_unit_1/add:z:0*
T0*0
_output_shapes
:�����������
.residual_unit_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp7residual_unit_2_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_2/conv2d_6/Conv2DConv2D$residual_unit_1/Relu_1:activations:06residual_unit_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_2/batch_normalization_5/ReadVariableOpReadVariableOp=residual_unit_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_5/ReadVariableOp_1ReadVariableOp?residual_unit_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3(residual_unit_2/conv2d_6/Conv2D:output:0<residual_unit_2/batch_normalization_5/ReadVariableOp:value:0>residual_unit_2/batch_normalization_5/ReadVariableOp_1:value:0Mresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_2/batch_normalization_5/AssignNewValueAssignVariableOpNresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceCresidual_unit_2/batch_normalization_5/FusedBatchNormV3:batch_mean:0F^residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_2/batch_normalization_5/AssignNewValue_1AssignVariableOpPresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_2/batch_normalization_5/FusedBatchNormV3:batch_variance:0H^residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_2/ReluRelu:residual_unit_2/batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
.residual_unit_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp7residual_unit_2_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_2/conv2d_7/Conv2DConv2D"residual_unit_2/Relu:activations:06residual_unit_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_2/batch_normalization_6/ReadVariableOpReadVariableOp=residual_unit_2_batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp?residual_unit_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3(residual_unit_2/conv2d_7/Conv2D:output:0<residual_unit_2/batch_normalization_6/ReadVariableOp:value:0>residual_unit_2/batch_normalization_6/ReadVariableOp_1:value:0Mresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_2/batch_normalization_6/AssignNewValueAssignVariableOpNresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceCresidual_unit_2/batch_normalization_6/FusedBatchNormV3:batch_mean:0F^residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_2/batch_normalization_6/AssignNewValue_1AssignVariableOpPresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_2/batch_normalization_6/FusedBatchNormV3:batch_variance:0H^residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_2/addAddV2:residual_unit_2/batch_normalization_6/FusedBatchNormV3:y:0$residual_unit_1/Relu_1:activations:0*
T0*0
_output_shapes
:����������r
residual_unit_2/Relu_1Reluresidual_unit_2/add:z:0*
T0*0
_output_shapes
:�����������
.residual_unit_3/conv2d_8/Conv2D/ReadVariableOpReadVariableOp7residual_unit_3_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_3/conv2d_8/Conv2DConv2D$residual_unit_2/Relu_1:activations:06residual_unit_3/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_3/batch_normalization_7/ReadVariableOpReadVariableOp=residual_unit_3_batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_7/ReadVariableOp_1ReadVariableOp?residual_unit_3_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(residual_unit_3/conv2d_8/Conv2D:output:0<residual_unit_3/batch_normalization_7/ReadVariableOp:value:0>residual_unit_3/batch_normalization_7/ReadVariableOp_1:value:0Mresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_3/batch_normalization_7/AssignNewValueAssignVariableOpNresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceCresidual_unit_3/batch_normalization_7/FusedBatchNormV3:batch_mean:0F^residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_3/batch_normalization_7/AssignNewValue_1AssignVariableOpPresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_3/batch_normalization_7/FusedBatchNormV3:batch_variance:0H^residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_3/ReluRelu:residual_unit_3/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
.residual_unit_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp7residual_unit_3_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_3/conv2d_9/Conv2DConv2D"residual_unit_3/Relu:activations:06residual_unit_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_3/batch_normalization_8/ReadVariableOpReadVariableOp=residual_unit_3_batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_8/ReadVariableOp_1ReadVariableOp?residual_unit_3_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3(residual_unit_3/conv2d_9/Conv2D:output:0<residual_unit_3/batch_normalization_8/ReadVariableOp:value:0>residual_unit_3/batch_normalization_8/ReadVariableOp_1:value:0Mresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_3/batch_normalization_8/AssignNewValueAssignVariableOpNresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceCresidual_unit_3/batch_normalization_8/FusedBatchNormV3:batch_mean:0F^residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_3/batch_normalization_8/AssignNewValue_1AssignVariableOpPresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_3/batch_normalization_8/FusedBatchNormV3:batch_variance:0H^residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
/residual_unit_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp8residual_unit_3_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_3/conv2d_10/Conv2DConv2D$residual_unit_2/Relu_1:activations:07residual_unit_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_3/batch_normalization_9/ReadVariableOpReadVariableOp=residual_unit_3_batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_9/ReadVariableOp_1ReadVariableOp?residual_unit_3_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)residual_unit_3/conv2d_10/Conv2D:output:0<residual_unit_3/batch_normalization_9/ReadVariableOp:value:0>residual_unit_3/batch_normalization_9/ReadVariableOp_1:value:0Mresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4residual_unit_3/batch_normalization_9/AssignNewValueAssignVariableOpNresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceCresidual_unit_3/batch_normalization_9/FusedBatchNormV3:batch_mean:0F^residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6residual_unit_3/batch_normalization_9/AssignNewValue_1AssignVariableOpPresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceGresidual_unit_3/batch_normalization_9/FusedBatchNormV3:batch_variance:0H^residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_3/addAddV2:residual_unit_3/batch_normalization_8/FusedBatchNormV3:y:0:residual_unit_3/batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������r
residual_unit_3/Relu_1Reluresidual_unit_3/add:z:0*
T0*0
_output_shapes
:�����������
/residual_unit_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp8residual_unit_4_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_4/conv2d_11/Conv2DConv2D$residual_unit_3/Relu_1:activations:07residual_unit_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_4/batch_normalization_10/ReadVariableOpReadVariableOp>residual_unit_4_batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_10/ReadVariableOp_1ReadVariableOp@residual_unit_4_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3)residual_unit_4/conv2d_11/Conv2D:output:0=residual_unit_4/batch_normalization_10/ReadVariableOp:value:0?residual_unit_4/batch_normalization_10/ReadVariableOp_1:value:0Nresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5residual_unit_4/batch_normalization_10/AssignNewValueAssignVariableOpOresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceDresidual_unit_4/batch_normalization_10/FusedBatchNormV3:batch_mean:0G^residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7residual_unit_4/batch_normalization_10/AssignNewValue_1AssignVariableOpQresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceHresidual_unit_4/batch_normalization_10/FusedBatchNormV3:batch_variance:0I^residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_4/ReluRelu;residual_unit_4/batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
/residual_unit_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp8residual_unit_4_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_4/conv2d_12/Conv2DConv2D"residual_unit_4/Relu:activations:07residual_unit_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_4/batch_normalization_11/ReadVariableOpReadVariableOp>residual_unit_4_batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_11/ReadVariableOp_1ReadVariableOp@residual_unit_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3)residual_unit_4/conv2d_12/Conv2D:output:0=residual_unit_4/batch_normalization_11/ReadVariableOp:value:0?residual_unit_4/batch_normalization_11/ReadVariableOp_1:value:0Nresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5residual_unit_4/batch_normalization_11/AssignNewValueAssignVariableOpOresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceDresidual_unit_4/batch_normalization_11/FusedBatchNormV3:batch_mean:0G^residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7residual_unit_4/batch_normalization_11/AssignNewValue_1AssignVariableOpQresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceHresidual_unit_4/batch_normalization_11/FusedBatchNormV3:batch_variance:0I^residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_4/addAddV2;residual_unit_4/batch_normalization_11/FusedBatchNormV3:y:0$residual_unit_3/Relu_1:activations:0*
T0*0
_output_shapes
:����������r
residual_unit_4/Relu_1Reluresidual_unit_4/add:z:0*
T0*0
_output_shapes
:�����������
/residual_unit_5/conv2d_13/Conv2D/ReadVariableOpReadVariableOp8residual_unit_5_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_5/conv2d_13/Conv2DConv2D$residual_unit_4/Relu_1:activations:07residual_unit_5/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_5/batch_normalization_12/ReadVariableOpReadVariableOp>residual_unit_5_batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_12/ReadVariableOp_1ReadVariableOp@residual_unit_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3)residual_unit_5/conv2d_13/Conv2D:output:0=residual_unit_5/batch_normalization_12/ReadVariableOp:value:0?residual_unit_5/batch_normalization_12/ReadVariableOp_1:value:0Nresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5residual_unit_5/batch_normalization_12/AssignNewValueAssignVariableOpOresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceDresidual_unit_5/batch_normalization_12/FusedBatchNormV3:batch_mean:0G^residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7residual_unit_5/batch_normalization_12/AssignNewValue_1AssignVariableOpQresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceHresidual_unit_5/batch_normalization_12/FusedBatchNormV3:batch_variance:0I^residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_5/ReluRelu;residual_unit_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
/residual_unit_5/conv2d_14/Conv2D/ReadVariableOpReadVariableOp8residual_unit_5_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_5/conv2d_14/Conv2DConv2D"residual_unit_5/Relu:activations:07residual_unit_5/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_5/batch_normalization_13/ReadVariableOpReadVariableOp>residual_unit_5_batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_13/ReadVariableOp_1ReadVariableOp@residual_unit_5_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3)residual_unit_5/conv2d_14/Conv2D:output:0=residual_unit_5/batch_normalization_13/ReadVariableOp:value:0?residual_unit_5/batch_normalization_13/ReadVariableOp_1:value:0Nresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5residual_unit_5/batch_normalization_13/AssignNewValueAssignVariableOpOresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceDresidual_unit_5/batch_normalization_13/FusedBatchNormV3:batch_mean:0G^residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7residual_unit_5/batch_normalization_13/AssignNewValue_1AssignVariableOpQresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resourceHresidual_unit_5/batch_normalization_13/FusedBatchNormV3:batch_variance:0I^residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
/residual_unit_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp8residual_unit_5_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_5/conv2d_15/Conv2DConv2D$residual_unit_4/Relu_1:activations:07residual_unit_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_5/batch_normalization_14/ReadVariableOpReadVariableOp>residual_unit_5_batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_14/ReadVariableOp_1ReadVariableOp@residual_unit_5_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3)residual_unit_5/conv2d_15/Conv2D:output:0=residual_unit_5/batch_normalization_14/ReadVariableOp:value:0?residual_unit_5/batch_normalization_14/ReadVariableOp_1:value:0Nresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5residual_unit_5/batch_normalization_14/AssignNewValueAssignVariableOpOresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resourceDresidual_unit_5/batch_normalization_14/FusedBatchNormV3:batch_mean:0G^residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7residual_unit_5/batch_normalization_14/AssignNewValue_1AssignVariableOpQresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resourceHresidual_unit_5/batch_normalization_14/FusedBatchNormV3:batch_variance:0I^residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
residual_unit_5/addAddV2;residual_unit_5/batch_normalization_13/FusedBatchNormV3:y:0;residual_unit_5/batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������r
residual_unit_5/Relu_1Reluresidual_unit_5/add:z:0*
T0*0
_output_shapes
:�����������
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d/MeanMean$residual_unit_5/Relu_1:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������V
sampling/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0�
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*(
_output_shapes
:�����������
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*(
_output_shapes
:����������W
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
sampling/truedivRealDivdense_1/BiasAdd:output:0sampling/truediv/y:output:0*
T0*(
_output_shapes
:����������\
sampling/ExpExpsampling/truediv:z:0*
T0*(
_output_shapes
:����������t
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*(
_output_shapes
:����������r
sampling/addAddV2sampling/mul:z:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������j

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������b

Identity_2Identitysampling/add:z:0^NoOp*
T0*(
_output_shapes
:�����������2
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^residual_unit/batch_normalization/AssignNewValue3^residual_unit/batch_normalization/AssignNewValue_1B^residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpD^residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_11^residual_unit/batch_normalization/ReadVariableOp3^residual_unit/batch_normalization/ReadVariableOp_13^residual_unit/batch_normalization_1/AssignNewValue5^residual_unit/batch_normalization_1/AssignNewValue_1D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_1/ReadVariableOp5^residual_unit/batch_normalization_1/ReadVariableOp_1-^residual_unit/conv2d_1/Conv2D/ReadVariableOp-^residual_unit/conv2d_2/Conv2D/ReadVariableOp5^residual_unit_1/batch_normalization_2/AssignNewValue7^residual_unit_1/batch_normalization_2/AssignNewValue_1F^residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpH^residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_15^residual_unit_1/batch_normalization_2/ReadVariableOp7^residual_unit_1/batch_normalization_2/ReadVariableOp_15^residual_unit_1/batch_normalization_3/AssignNewValue7^residual_unit_1/batch_normalization_3/AssignNewValue_1F^residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpH^residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_15^residual_unit_1/batch_normalization_3/ReadVariableOp7^residual_unit_1/batch_normalization_3/ReadVariableOp_15^residual_unit_1/batch_normalization_4/AssignNewValue7^residual_unit_1/batch_normalization_4/AssignNewValue_1F^residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpH^residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_15^residual_unit_1/batch_normalization_4/ReadVariableOp7^residual_unit_1/batch_normalization_4/ReadVariableOp_1/^residual_unit_1/conv2d_3/Conv2D/ReadVariableOp/^residual_unit_1/conv2d_4/Conv2D/ReadVariableOp/^residual_unit_1/conv2d_5/Conv2D/ReadVariableOp5^residual_unit_2/batch_normalization_5/AssignNewValue7^residual_unit_2/batch_normalization_5/AssignNewValue_1F^residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpH^residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_15^residual_unit_2/batch_normalization_5/ReadVariableOp7^residual_unit_2/batch_normalization_5/ReadVariableOp_15^residual_unit_2/batch_normalization_6/AssignNewValue7^residual_unit_2/batch_normalization_6/AssignNewValue_1F^residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_15^residual_unit_2/batch_normalization_6/ReadVariableOp7^residual_unit_2/batch_normalization_6/ReadVariableOp_1/^residual_unit_2/conv2d_6/Conv2D/ReadVariableOp/^residual_unit_2/conv2d_7/Conv2D/ReadVariableOp5^residual_unit_3/batch_normalization_7/AssignNewValue7^residual_unit_3/batch_normalization_7/AssignNewValue_1F^residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpH^residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_15^residual_unit_3/batch_normalization_7/ReadVariableOp7^residual_unit_3/batch_normalization_7/ReadVariableOp_15^residual_unit_3/batch_normalization_8/AssignNewValue7^residual_unit_3/batch_normalization_8/AssignNewValue_1F^residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpH^residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_15^residual_unit_3/batch_normalization_8/ReadVariableOp7^residual_unit_3/batch_normalization_8/ReadVariableOp_15^residual_unit_3/batch_normalization_9/AssignNewValue7^residual_unit_3/batch_normalization_9/AssignNewValue_1F^residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpH^residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_15^residual_unit_3/batch_normalization_9/ReadVariableOp7^residual_unit_3/batch_normalization_9/ReadVariableOp_10^residual_unit_3/conv2d_10/Conv2D/ReadVariableOp/^residual_unit_3/conv2d_8/Conv2D/ReadVariableOp/^residual_unit_3/conv2d_9/Conv2D/ReadVariableOp6^residual_unit_4/batch_normalization_10/AssignNewValue8^residual_unit_4/batch_normalization_10/AssignNewValue_1G^residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpI^residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_16^residual_unit_4/batch_normalization_10/ReadVariableOp8^residual_unit_4/batch_normalization_10/ReadVariableOp_16^residual_unit_4/batch_normalization_11/AssignNewValue8^residual_unit_4/batch_normalization_11/AssignNewValue_1G^residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpI^residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_16^residual_unit_4/batch_normalization_11/ReadVariableOp8^residual_unit_4/batch_normalization_11/ReadVariableOp_10^residual_unit_4/conv2d_11/Conv2D/ReadVariableOp0^residual_unit_4/conv2d_12/Conv2D/ReadVariableOp6^residual_unit_5/batch_normalization_12/AssignNewValue8^residual_unit_5/batch_normalization_12/AssignNewValue_1G^residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpI^residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_16^residual_unit_5/batch_normalization_12/ReadVariableOp8^residual_unit_5/batch_normalization_12/ReadVariableOp_16^residual_unit_5/batch_normalization_13/AssignNewValue8^residual_unit_5/batch_normalization_13/AssignNewValue_1G^residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpI^residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_16^residual_unit_5/batch_normalization_13/ReadVariableOp8^residual_unit_5/batch_normalization_13/ReadVariableOp_16^residual_unit_5/batch_normalization_14/AssignNewValue8^residual_unit_5/batch_normalization_14/AssignNewValue_1G^residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpI^residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_16^residual_unit_5/batch_normalization_14/ReadVariableOp8^residual_unit_5/batch_normalization_14/ReadVariableOp_10^residual_unit_5/conv2d_13/Conv2D/ReadVariableOp0^residual_unit_5/conv2d_14/Conv2D/ReadVariableOp0^residual_unit_5/conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0residual_unit/batch_normalization/AssignNewValue0residual_unit/batch_normalization/AssignNewValue2h
2residual_unit/batch_normalization/AssignNewValue_12residual_unit/batch_normalization/AssignNewValue_12�
Aresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpAresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_12d
0residual_unit/batch_normalization/ReadVariableOp0residual_unit/batch_normalization/ReadVariableOp2h
2residual_unit/batch_normalization/ReadVariableOp_12residual_unit/batch_normalization/ReadVariableOp_12h
2residual_unit/batch_normalization_1/AssignNewValue2residual_unit/batch_normalization_1/AssignNewValue2l
4residual_unit/batch_normalization_1/AssignNewValue_14residual_unit/batch_normalization_1/AssignNewValue_12�
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_1/ReadVariableOp2residual_unit/batch_normalization_1/ReadVariableOp2l
4residual_unit/batch_normalization_1/ReadVariableOp_14residual_unit/batch_normalization_1/ReadVariableOp_12\
,residual_unit/conv2d_1/Conv2D/ReadVariableOp,residual_unit/conv2d_1/Conv2D/ReadVariableOp2\
,residual_unit/conv2d_2/Conv2D/ReadVariableOp,residual_unit/conv2d_2/Conv2D/ReadVariableOp2l
4residual_unit_1/batch_normalization_2/AssignNewValue4residual_unit_1/batch_normalization_2/AssignNewValue2p
6residual_unit_1/batch_normalization_2/AssignNewValue_16residual_unit_1/batch_normalization_2/AssignNewValue_12�
Eresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpEresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_2/ReadVariableOp4residual_unit_1/batch_normalization_2/ReadVariableOp2p
6residual_unit_1/batch_normalization_2/ReadVariableOp_16residual_unit_1/batch_normalization_2/ReadVariableOp_12l
4residual_unit_1/batch_normalization_3/AssignNewValue4residual_unit_1/batch_normalization_3/AssignNewValue2p
6residual_unit_1/batch_normalization_3/AssignNewValue_16residual_unit_1/batch_normalization_3/AssignNewValue_12�
Eresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpEresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_3/ReadVariableOp4residual_unit_1/batch_normalization_3/ReadVariableOp2p
6residual_unit_1/batch_normalization_3/ReadVariableOp_16residual_unit_1/batch_normalization_3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_4/AssignNewValue4residual_unit_1/batch_normalization_4/AssignNewValue2p
6residual_unit_1/batch_normalization_4/AssignNewValue_16residual_unit_1/batch_normalization_4/AssignNewValue_12�
Eresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpEresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_4/ReadVariableOp4residual_unit_1/batch_normalization_4/ReadVariableOp2p
6residual_unit_1/batch_normalization_4/ReadVariableOp_16residual_unit_1/batch_normalization_4/ReadVariableOp_12`
.residual_unit_1/conv2d_3/Conv2D/ReadVariableOp.residual_unit_1/conv2d_3/Conv2D/ReadVariableOp2`
.residual_unit_1/conv2d_4/Conv2D/ReadVariableOp.residual_unit_1/conv2d_4/Conv2D/ReadVariableOp2`
.residual_unit_1/conv2d_5/Conv2D/ReadVariableOp.residual_unit_1/conv2d_5/Conv2D/ReadVariableOp2l
4residual_unit_2/batch_normalization_5/AssignNewValue4residual_unit_2/batch_normalization_5/AssignNewValue2p
6residual_unit_2/batch_normalization_5/AssignNewValue_16residual_unit_2/batch_normalization_5/AssignNewValue_12�
Eresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpEresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_2/batch_normalization_5/ReadVariableOp4residual_unit_2/batch_normalization_5/ReadVariableOp2p
6residual_unit_2/batch_normalization_5/ReadVariableOp_16residual_unit_2/batch_normalization_5/ReadVariableOp_12l
4residual_unit_2/batch_normalization_6/AssignNewValue4residual_unit_2/batch_normalization_6/AssignNewValue2p
6residual_unit_2/batch_normalization_6/AssignNewValue_16residual_unit_2/batch_normalization_6/AssignNewValue_12�
Eresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_2/batch_normalization_6/ReadVariableOp4residual_unit_2/batch_normalization_6/ReadVariableOp2p
6residual_unit_2/batch_normalization_6/ReadVariableOp_16residual_unit_2/batch_normalization_6/ReadVariableOp_12`
.residual_unit_2/conv2d_6/Conv2D/ReadVariableOp.residual_unit_2/conv2d_6/Conv2D/ReadVariableOp2`
.residual_unit_2/conv2d_7/Conv2D/ReadVariableOp.residual_unit_2/conv2d_7/Conv2D/ReadVariableOp2l
4residual_unit_3/batch_normalization_7/AssignNewValue4residual_unit_3/batch_normalization_7/AssignNewValue2p
6residual_unit_3/batch_normalization_7/AssignNewValue_16residual_unit_3/batch_normalization_7/AssignNewValue_12�
Eresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpEresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_3/batch_normalization_7/ReadVariableOp4residual_unit_3/batch_normalization_7/ReadVariableOp2p
6residual_unit_3/batch_normalization_7/ReadVariableOp_16residual_unit_3/batch_normalization_7/ReadVariableOp_12l
4residual_unit_3/batch_normalization_8/AssignNewValue4residual_unit_3/batch_normalization_8/AssignNewValue2p
6residual_unit_3/batch_normalization_8/AssignNewValue_16residual_unit_3/batch_normalization_8/AssignNewValue_12�
Eresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpEresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_3/batch_normalization_8/ReadVariableOp4residual_unit_3/batch_normalization_8/ReadVariableOp2p
6residual_unit_3/batch_normalization_8/ReadVariableOp_16residual_unit_3/batch_normalization_8/ReadVariableOp_12l
4residual_unit_3/batch_normalization_9/AssignNewValue4residual_unit_3/batch_normalization_9/AssignNewValue2p
6residual_unit_3/batch_normalization_9/AssignNewValue_16residual_unit_3/batch_normalization_9/AssignNewValue_12�
Eresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpEresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_3/batch_normalization_9/ReadVariableOp4residual_unit_3/batch_normalization_9/ReadVariableOp2p
6residual_unit_3/batch_normalization_9/ReadVariableOp_16residual_unit_3/batch_normalization_9/ReadVariableOp_12b
/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp2`
.residual_unit_3/conv2d_8/Conv2D/ReadVariableOp.residual_unit_3/conv2d_8/Conv2D/ReadVariableOp2`
.residual_unit_3/conv2d_9/Conv2D/ReadVariableOp.residual_unit_3/conv2d_9/Conv2D/ReadVariableOp2n
5residual_unit_4/batch_normalization_10/AssignNewValue5residual_unit_4/batch_normalization_10/AssignNewValue2r
7residual_unit_4/batch_normalization_10/AssignNewValue_17residual_unit_4/batch_normalization_10/AssignNewValue_12�
Fresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpFresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_4/batch_normalization_10/ReadVariableOp5residual_unit_4/batch_normalization_10/ReadVariableOp2r
7residual_unit_4/batch_normalization_10/ReadVariableOp_17residual_unit_4/batch_normalization_10/ReadVariableOp_12n
5residual_unit_4/batch_normalization_11/AssignNewValue5residual_unit_4/batch_normalization_11/AssignNewValue2r
7residual_unit_4/batch_normalization_11/AssignNewValue_17residual_unit_4/batch_normalization_11/AssignNewValue_12�
Fresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpFresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_4/batch_normalization_11/ReadVariableOp5residual_unit_4/batch_normalization_11/ReadVariableOp2r
7residual_unit_4/batch_normalization_11/ReadVariableOp_17residual_unit_4/batch_normalization_11/ReadVariableOp_12b
/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp2b
/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp2n
5residual_unit_5/batch_normalization_12/AssignNewValue5residual_unit_5/batch_normalization_12/AssignNewValue2r
7residual_unit_5/batch_normalization_12/AssignNewValue_17residual_unit_5/batch_normalization_12/AssignNewValue_12�
Fresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpFresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_5/batch_normalization_12/ReadVariableOp5residual_unit_5/batch_normalization_12/ReadVariableOp2r
7residual_unit_5/batch_normalization_12/ReadVariableOp_17residual_unit_5/batch_normalization_12/ReadVariableOp_12n
5residual_unit_5/batch_normalization_13/AssignNewValue5residual_unit_5/batch_normalization_13/AssignNewValue2r
7residual_unit_5/batch_normalization_13/AssignNewValue_17residual_unit_5/batch_normalization_13/AssignNewValue_12�
Fresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpFresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_5/batch_normalization_13/ReadVariableOp5residual_unit_5/batch_normalization_13/ReadVariableOp2r
7residual_unit_5/batch_normalization_13/ReadVariableOp_17residual_unit_5/batch_normalization_13/ReadVariableOp_12n
5residual_unit_5/batch_normalization_14/AssignNewValue5residual_unit_5/batch_normalization_14/AssignNewValue2r
7residual_unit_5/batch_normalization_14/AssignNewValue_17residual_unit_5/batch_normalization_14/AssignNewValue_12�
Fresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpFresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_5/batch_normalization_14/ReadVariableOp5residual_unit_5/batch_normalization_14/ReadVariableOp2r
7residual_unit_5/batch_normalization_14/ReadVariableOp_17residual_unit_5/batch_normalization_14/ReadVariableOp_12b
/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp2b
/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp2b
/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�]
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022632

inputsD
(conv2d_13_conv2d_readvariableop_resource:��=
.batch_normalization_12_readvariableop_resource:	�?
0batch_normalization_12_readvariableop_1_resource:	�N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_14_conv2d_readvariableop_resource:��=
.batch_normalization_13_readvariableop_resource:	�?
0batch_normalization_13_readvariableop_1_resource:	�N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_15_conv2d_readvariableop_resource:��=
.batch_normalization_14_readvariableop_resource:	�?
0batch_normalization_14_readvariableop_1_resource:	�N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_12/AssignNewValue�'batch_normalization_12/AssignNewValue_1�6batch_normalization_12/FusedBatchNormV3/ReadVariableOp�8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_12/ReadVariableOp�'batch_normalization_12/ReadVariableOp_1�%batch_normalization_13/AssignNewValue�'batch_normalization_13/AssignNewValue_1�6batch_normalization_13/FusedBatchNormV3/ReadVariableOp�8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_13/ReadVariableOp�'batch_normalization_13/ReadVariableOp_1�%batch_normalization_14/AssignNewValue�'batch_normalization_14/AssignNewValue_1�6batch_normalization_14/FusedBatchNormV3/ReadVariableOp�8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_14/ReadVariableOp�'batch_normalization_14/ReadVariableOp_1�conv2d_13/Conv2D/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_13/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_14/Conv2DConv2DRelu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_14/Conv2D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_15/Conv2D:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
addAddV2+batch_normalization_13/FusedBatchNormV3:y:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1 ^conv2d_13/Conv2D/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�@
%__inference__traced_restore_128024159
file_prefix8
assignvariableop_conv2d_kernel:@3
assignvariableop_1_dense_kernel:
��,
assignvariableop_2_dense_bias:	�5
!assignvariableop_3_dense_1_kernel:
��.
assignvariableop_4_dense_1_bias:	�J
0assignvariableop_5_residual_unit_conv2d_1_kernel:@@H
:assignvariableop_6_residual_unit_batch_normalization_gamma:@G
9assignvariableop_7_residual_unit_batch_normalization_beta:@J
0assignvariableop_8_residual_unit_conv2d_2_kernel:@@J
<assignvariableop_9_residual_unit_batch_normalization_1_gamma:@J
<assignvariableop_10_residual_unit_batch_normalization_1_beta:@O
Aassignvariableop_11_residual_unit_batch_normalization_moving_mean:@S
Eassignvariableop_12_residual_unit_batch_normalization_moving_variance:@Q
Cassignvariableop_13_residual_unit_batch_normalization_1_moving_mean:@U
Gassignvariableop_14_residual_unit_batch_normalization_1_moving_variance:@N
3assignvariableop_15_residual_unit_1_conv2d_3_kernel:@�N
?assignvariableop_16_residual_unit_1_batch_normalization_2_gamma:	�M
>assignvariableop_17_residual_unit_1_batch_normalization_2_beta:	�O
3assignvariableop_18_residual_unit_1_conv2d_4_kernel:��N
?assignvariableop_19_residual_unit_1_batch_normalization_3_gamma:	�M
>assignvariableop_20_residual_unit_1_batch_normalization_3_beta:	�N
3assignvariableop_21_residual_unit_1_conv2d_5_kernel:@�N
?assignvariableop_22_residual_unit_1_batch_normalization_4_gamma:	�M
>assignvariableop_23_residual_unit_1_batch_normalization_4_beta:	�T
Eassignvariableop_24_residual_unit_1_batch_normalization_2_moving_mean:	�X
Iassignvariableop_25_residual_unit_1_batch_normalization_2_moving_variance:	�T
Eassignvariableop_26_residual_unit_1_batch_normalization_3_moving_mean:	�X
Iassignvariableop_27_residual_unit_1_batch_normalization_3_moving_variance:	�T
Eassignvariableop_28_residual_unit_1_batch_normalization_4_moving_mean:	�X
Iassignvariableop_29_residual_unit_1_batch_normalization_4_moving_variance:	�O
3assignvariableop_30_residual_unit_2_conv2d_6_kernel:��N
?assignvariableop_31_residual_unit_2_batch_normalization_5_gamma:	�M
>assignvariableop_32_residual_unit_2_batch_normalization_5_beta:	�O
3assignvariableop_33_residual_unit_2_conv2d_7_kernel:��N
?assignvariableop_34_residual_unit_2_batch_normalization_6_gamma:	�M
>assignvariableop_35_residual_unit_2_batch_normalization_6_beta:	�T
Eassignvariableop_36_residual_unit_2_batch_normalization_5_moving_mean:	�X
Iassignvariableop_37_residual_unit_2_batch_normalization_5_moving_variance:	�T
Eassignvariableop_38_residual_unit_2_batch_normalization_6_moving_mean:	�X
Iassignvariableop_39_residual_unit_2_batch_normalization_6_moving_variance:	�O
3assignvariableop_40_residual_unit_3_conv2d_8_kernel:��N
?assignvariableop_41_residual_unit_3_batch_normalization_7_gamma:	�M
>assignvariableop_42_residual_unit_3_batch_normalization_7_beta:	�O
3assignvariableop_43_residual_unit_3_conv2d_9_kernel:��N
?assignvariableop_44_residual_unit_3_batch_normalization_8_gamma:	�M
>assignvariableop_45_residual_unit_3_batch_normalization_8_beta:	�P
4assignvariableop_46_residual_unit_3_conv2d_10_kernel:��N
?assignvariableop_47_residual_unit_3_batch_normalization_9_gamma:	�M
>assignvariableop_48_residual_unit_3_batch_normalization_9_beta:	�T
Eassignvariableop_49_residual_unit_3_batch_normalization_7_moving_mean:	�X
Iassignvariableop_50_residual_unit_3_batch_normalization_7_moving_variance:	�T
Eassignvariableop_51_residual_unit_3_batch_normalization_8_moving_mean:	�X
Iassignvariableop_52_residual_unit_3_batch_normalization_8_moving_variance:	�T
Eassignvariableop_53_residual_unit_3_batch_normalization_9_moving_mean:	�X
Iassignvariableop_54_residual_unit_3_batch_normalization_9_moving_variance:	�P
4assignvariableop_55_residual_unit_4_conv2d_11_kernel:��O
@assignvariableop_56_residual_unit_4_batch_normalization_10_gamma:	�N
?assignvariableop_57_residual_unit_4_batch_normalization_10_beta:	�P
4assignvariableop_58_residual_unit_4_conv2d_12_kernel:��O
@assignvariableop_59_residual_unit_4_batch_normalization_11_gamma:	�N
?assignvariableop_60_residual_unit_4_batch_normalization_11_beta:	�U
Fassignvariableop_61_residual_unit_4_batch_normalization_10_moving_mean:	�Y
Jassignvariableop_62_residual_unit_4_batch_normalization_10_moving_variance:	�U
Fassignvariableop_63_residual_unit_4_batch_normalization_11_moving_mean:	�Y
Jassignvariableop_64_residual_unit_4_batch_normalization_11_moving_variance:	�P
4assignvariableop_65_residual_unit_5_conv2d_13_kernel:��O
@assignvariableop_66_residual_unit_5_batch_normalization_12_gamma:	�N
?assignvariableop_67_residual_unit_5_batch_normalization_12_beta:	�P
4assignvariableop_68_residual_unit_5_conv2d_14_kernel:��O
@assignvariableop_69_residual_unit_5_batch_normalization_13_gamma:	�N
?assignvariableop_70_residual_unit_5_batch_normalization_13_beta:	�P
4assignvariableop_71_residual_unit_5_conv2d_15_kernel:��O
@assignvariableop_72_residual_unit_5_batch_normalization_14_gamma:	�N
?assignvariableop_73_residual_unit_5_batch_normalization_14_beta:	�U
Fassignvariableop_74_residual_unit_5_batch_normalization_12_moving_mean:	�Y
Jassignvariableop_75_residual_unit_5_batch_normalization_12_moving_variance:	�U
Fassignvariableop_76_residual_unit_5_batch_normalization_13_moving_mean:	�Y
Jassignvariableop_77_residual_unit_5_batch_normalization_13_moving_variance:	�U
Fassignvariableop_78_residual_unit_5_batch_normalization_14_moving_mean:	�Y
Jassignvariableop_79_residual_unit_5_batch_normalization_14_moving_variance:	�
identity_81��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�
value�B�QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�
value�B�QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_residual_unit_conv2d_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp:assignvariableop_6_residual_unit_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp9assignvariableop_7_residual_unit_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_residual_unit_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp<assignvariableop_9_residual_unit_batch_normalization_1_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp<assignvariableop_10_residual_unit_batch_normalization_1_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpAassignvariableop_11_residual_unit_batch_normalization_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpEassignvariableop_12_residual_unit_batch_normalization_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpCassignvariableop_13_residual_unit_batch_normalization_1_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpGassignvariableop_14_residual_unit_batch_normalization_1_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp3assignvariableop_15_residual_unit_1_conv2d_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp?assignvariableop_16_residual_unit_1_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp>assignvariableop_17_residual_unit_1_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_residual_unit_1_conv2d_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp?assignvariableop_19_residual_unit_1_batch_normalization_3_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp>assignvariableop_20_residual_unit_1_batch_normalization_3_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp3assignvariableop_21_residual_unit_1_conv2d_5_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp?assignvariableop_22_residual_unit_1_batch_normalization_4_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp>assignvariableop_23_residual_unit_1_batch_normalization_4_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpEassignvariableop_24_residual_unit_1_batch_normalization_2_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpIassignvariableop_25_residual_unit_1_batch_normalization_2_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpEassignvariableop_26_residual_unit_1_batch_normalization_3_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpIassignvariableop_27_residual_unit_1_batch_normalization_3_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpEassignvariableop_28_residual_unit_1_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpIassignvariableop_29_residual_unit_1_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp3assignvariableop_30_residual_unit_2_conv2d_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp?assignvariableop_31_residual_unit_2_batch_normalization_5_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp>assignvariableop_32_residual_unit_2_batch_normalization_5_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp3assignvariableop_33_residual_unit_2_conv2d_7_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp?assignvariableop_34_residual_unit_2_batch_normalization_6_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp>assignvariableop_35_residual_unit_2_batch_normalization_6_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpEassignvariableop_36_residual_unit_2_batch_normalization_5_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpIassignvariableop_37_residual_unit_2_batch_normalization_5_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpEassignvariableop_38_residual_unit_2_batch_normalization_6_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpIassignvariableop_39_residual_unit_2_batch_normalization_6_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp3assignvariableop_40_residual_unit_3_conv2d_8_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp?assignvariableop_41_residual_unit_3_batch_normalization_7_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp>assignvariableop_42_residual_unit_3_batch_normalization_7_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp3assignvariableop_43_residual_unit_3_conv2d_9_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp?assignvariableop_44_residual_unit_3_batch_normalization_8_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp>assignvariableop_45_residual_unit_3_batch_normalization_8_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp4assignvariableop_46_residual_unit_3_conv2d_10_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp?assignvariableop_47_residual_unit_3_batch_normalization_9_gammaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp>assignvariableop_48_residual_unit_3_batch_normalization_9_betaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpEassignvariableop_49_residual_unit_3_batch_normalization_7_moving_meanIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpIassignvariableop_50_residual_unit_3_batch_normalization_7_moving_varianceIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpEassignvariableop_51_residual_unit_3_batch_normalization_8_moving_meanIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpIassignvariableop_52_residual_unit_3_batch_normalization_8_moving_varianceIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpEassignvariableop_53_residual_unit_3_batch_normalization_9_moving_meanIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpIassignvariableop_54_residual_unit_3_batch_normalization_9_moving_varianceIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp4assignvariableop_55_residual_unit_4_conv2d_11_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp@assignvariableop_56_residual_unit_4_batch_normalization_10_gammaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp?assignvariableop_57_residual_unit_4_batch_normalization_10_betaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp4assignvariableop_58_residual_unit_4_conv2d_12_kernelIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp@assignvariableop_59_residual_unit_4_batch_normalization_11_gammaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp?assignvariableop_60_residual_unit_4_batch_normalization_11_betaIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpFassignvariableop_61_residual_unit_4_batch_normalization_10_moving_meanIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpJassignvariableop_62_residual_unit_4_batch_normalization_10_moving_varianceIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpFassignvariableop_63_residual_unit_4_batch_normalization_11_moving_meanIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpJassignvariableop_64_residual_unit_4_batch_normalization_11_moving_varianceIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp4assignvariableop_65_residual_unit_5_conv2d_13_kernelIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp@assignvariableop_66_residual_unit_5_batch_normalization_12_gammaIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp?assignvariableop_67_residual_unit_5_batch_normalization_12_betaIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp4assignvariableop_68_residual_unit_5_conv2d_14_kernelIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp@assignvariableop_69_residual_unit_5_batch_normalization_13_gammaIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp?assignvariableop_70_residual_unit_5_batch_normalization_13_betaIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp4assignvariableop_71_residual_unit_5_conv2d_15_kernelIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp@assignvariableop_72_residual_unit_5_batch_normalization_14_gammaIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp?assignvariableop_73_residual_unit_5_batch_normalization_14_betaIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpFassignvariableop_74_residual_unit_5_batch_normalization_12_moving_meanIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpJassignvariableop_75_residual_unit_5_batch_normalization_12_moving_varianceIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpFassignvariableop_76_residual_unit_5_batch_normalization_13_moving_meanIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpJassignvariableop_77_residual_unit_5_batch_normalization_13_moving_varianceIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpFassignvariableop_78_residual_unit_5_batch_normalization_14_moving_meanIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpJassignvariableop_79_residual_unit_5_batch_normalization_14_moving_varianceIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_80Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_81IdentityIdentity_80:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_81Identity_81:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
:__inference_batch_normalization_11_layer_call_fn_128023422

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128017826�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_layer_call_fn_128022740

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128017122�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022758

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
G
+__inference_flatten_layer_call_fn_128022648

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_128018534a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_10_layer_call_fn_128023347

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128017731�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023148

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_5_layer_call_fn_128023050

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128017442�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022962

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_11_layer_call_fn_128023409

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128017795�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128022643

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128018039

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�G
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022574

inputsD
(conv2d_13_conv2d_readvariableop_resource:��=
.batch_normalization_12_readvariableop_resource:	�?
0batch_normalization_12_readvariableop_1_resource:	�N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_14_conv2d_readvariableop_resource:��=
.batch_normalization_13_readvariableop_resource:	�?
0batch_normalization_13_readvariableop_1_resource:	�N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_15_conv2d_readvariableop_resource:��=
.batch_normalization_14_readvariableop_resource:	�?
0batch_normalization_14_readvariableop_1_resource:	�N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_12/FusedBatchNormV3/ReadVariableOp�8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_12/ReadVariableOp�'batch_normalization_12/ReadVariableOp_1�6batch_normalization_13/FusedBatchNormV3/ReadVariableOp�8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_13/ReadVariableOp�'batch_normalization_13/ReadVariableOp_1�6batch_normalization_14/FusedBatchNormV3/ReadVariableOp�8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_14/ReadVariableOp�'batch_normalization_14/ReadVariableOp_1�conv2d_13/Conv2D/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_13/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( t
ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_14/Conv2DConv2DRelu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_14/Conv2D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_15/Conv2D:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
addAddV2+batch_normalization_13/FusedBatchNormV3:y:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1 ^conv2d_13/Conv2D/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
t
G__inference_sampling_layer_call_and_return_conditional_losses_128018584

inputs
inputs_1
identity�=
ShapeShapeinputs_1*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*(
_output_shapes
:����������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:����������}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:����������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
truedivRealDivinputs_1truediv/y:output:0*
T0*(
_output_shapes
:����������J
ExpExptruediv:z:0*
T0*(
_output_shapes
:����������Y
mulMulrandom_normal:z:0Exp:y:0*
T0*(
_output_shapes
:����������P
addAddV2mul:z:0inputs*
T0*(
_output_shapes
:����������P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128017570

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_3_layer_call_fn_128022926

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128017314�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�[
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128019106

inputsC
'conv2d_8_conv2d_readvariableop_resource:��<
-batch_normalization_7_readvariableop_resource:	�>
/batch_normalization_7_readvariableop_1_resource:	�M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��<
-batch_normalization_8_readvariableop_resource:	�>
/batch_normalization_8_readvariableop_1_resource:	�M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_7/AssignNewValue�&batch_normalization_7/AssignNewValue_1�5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�$batch_normalization_8/AssignNewValue�&batch_normalization_8/AssignNewValue_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�conv2d_10/Conv2D/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2DConv2DRelu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_9/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_10/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
addAddV2*batch_normalization_8/FusedBatchNormV3:y:0*batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_10/Conv2D/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�b
$__inference__wrapped_model_128017057
input_1E
+model_conv2d_conv2d_readvariableop_resource:@U
;model_residual_unit_conv2d_1_conv2d_readvariableop_resource:@@M
?model_residual_unit_batch_normalization_readvariableop_resource:@O
Amodel_residual_unit_batch_normalization_readvariableop_1_resource:@^
Pmodel_residual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource:@`
Rmodel_residual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@U
;model_residual_unit_conv2d_2_conv2d_readvariableop_resource:@@O
Amodel_residual_unit_batch_normalization_1_readvariableop_resource:@Q
Cmodel_residual_unit_batch_normalization_1_readvariableop_1_resource:@`
Rmodel_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@b
Tmodel_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@X
=model_residual_unit_1_conv2d_3_conv2d_readvariableop_resource:@�R
Cmodel_residual_unit_1_batch_normalization_2_readvariableop_resource:	�T
Emodel_residual_unit_1_batch_normalization_2_readvariableop_1_resource:	�c
Tmodel_residual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�Y
=model_residual_unit_1_conv2d_4_conv2d_readvariableop_resource:��R
Cmodel_residual_unit_1_batch_normalization_3_readvariableop_resource:	�T
Emodel_residual_unit_1_batch_normalization_3_readvariableop_1_resource:	�c
Tmodel_residual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�X
=model_residual_unit_1_conv2d_5_conv2d_readvariableop_resource:@�R
Cmodel_residual_unit_1_batch_normalization_4_readvariableop_resource:	�T
Emodel_residual_unit_1_batch_normalization_4_readvariableop_1_resource:	�c
Tmodel_residual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�Y
=model_residual_unit_2_conv2d_6_conv2d_readvariableop_resource:��R
Cmodel_residual_unit_2_batch_normalization_5_readvariableop_resource:	�T
Emodel_residual_unit_2_batch_normalization_5_readvariableop_1_resource:	�c
Tmodel_residual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�Y
=model_residual_unit_2_conv2d_7_conv2d_readvariableop_resource:��R
Cmodel_residual_unit_2_batch_normalization_6_readvariableop_resource:	�T
Emodel_residual_unit_2_batch_normalization_6_readvariableop_1_resource:	�c
Tmodel_residual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�Y
=model_residual_unit_3_conv2d_8_conv2d_readvariableop_resource:��R
Cmodel_residual_unit_3_batch_normalization_7_readvariableop_resource:	�T
Emodel_residual_unit_3_batch_normalization_7_readvariableop_1_resource:	�c
Tmodel_residual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�Y
=model_residual_unit_3_conv2d_9_conv2d_readvariableop_resource:��R
Cmodel_residual_unit_3_batch_normalization_8_readvariableop_resource:	�T
Emodel_residual_unit_3_batch_normalization_8_readvariableop_1_resource:	�c
Tmodel_residual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�Z
>model_residual_unit_3_conv2d_10_conv2d_readvariableop_resource:��R
Cmodel_residual_unit_3_batch_normalization_9_readvariableop_resource:	�T
Emodel_residual_unit_3_batch_normalization_9_readvariableop_1_resource:	�c
Tmodel_residual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�e
Vmodel_residual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�Z
>model_residual_unit_4_conv2d_11_conv2d_readvariableop_resource:��S
Dmodel_residual_unit_4_batch_normalization_10_readvariableop_resource:	�U
Fmodel_residual_unit_4_batch_normalization_10_readvariableop_1_resource:	�d
Umodel_residual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�f
Wmodel_residual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�Z
>model_residual_unit_4_conv2d_12_conv2d_readvariableop_resource:��S
Dmodel_residual_unit_4_batch_normalization_11_readvariableop_resource:	�U
Fmodel_residual_unit_4_batch_normalization_11_readvariableop_1_resource:	�d
Umodel_residual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�f
Wmodel_residual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�Z
>model_residual_unit_5_conv2d_13_conv2d_readvariableop_resource:��S
Dmodel_residual_unit_5_batch_normalization_12_readvariableop_resource:	�U
Fmodel_residual_unit_5_batch_normalization_12_readvariableop_1_resource:	�d
Umodel_residual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�f
Wmodel_residual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�Z
>model_residual_unit_5_conv2d_14_conv2d_readvariableop_resource:��S
Dmodel_residual_unit_5_batch_normalization_13_readvariableop_resource:	�U
Fmodel_residual_unit_5_batch_normalization_13_readvariableop_1_resource:	�d
Umodel_residual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�f
Wmodel_residual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�Z
>model_residual_unit_5_conv2d_15_conv2d_readvariableop_resource:��S
Dmodel_residual_unit_5_batch_normalization_14_readvariableop_resource:	�U
Fmodel_residual_unit_5_batch_normalization_14_readvariableop_1_resource:	�d
Umodel_residual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�f
Wmodel_residual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�>
*model_dense_matmul_readvariableop_resource:
��:
+model_dense_biasadd_readvariableop_resource:	�@
,model_dense_1_matmul_readvariableop_resource:
��<
-model_dense_1_biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��"model/conv2d/Conv2D/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�Gmodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp�Imodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�6model/residual_unit/batch_normalization/ReadVariableOp�8model/residual_unit/batch_normalization/ReadVariableOp_1�Imodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Kmodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�8model/residual_unit/batch_normalization_1/ReadVariableOp�:model/residual_unit/batch_normalization_1/ReadVariableOp_1�2model/residual_unit/conv2d_1/Conv2D/ReadVariableOp�2model/residual_unit/conv2d_2/Conv2D/ReadVariableOp�Kmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_1/batch_normalization_2/ReadVariableOp�<model/residual_unit_1/batch_normalization_2/ReadVariableOp_1�Kmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_1/batch_normalization_3/ReadVariableOp�<model/residual_unit_1/batch_normalization_3/ReadVariableOp_1�Kmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_1/batch_normalization_4/ReadVariableOp�<model/residual_unit_1/batch_normalization_4/ReadVariableOp_1�4model/residual_unit_1/conv2d_3/Conv2D/ReadVariableOp�4model/residual_unit_1/conv2d_4/Conv2D/ReadVariableOp�4model/residual_unit_1/conv2d_5/Conv2D/ReadVariableOp�Kmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_2/batch_normalization_5/ReadVariableOp�<model/residual_unit_2/batch_normalization_5/ReadVariableOp_1�Kmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_2/batch_normalization_6/ReadVariableOp�<model/residual_unit_2/batch_normalization_6/ReadVariableOp_1�4model/residual_unit_2/conv2d_6/Conv2D/ReadVariableOp�4model/residual_unit_2/conv2d_7/Conv2D/ReadVariableOp�Kmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_3/batch_normalization_7/ReadVariableOp�<model/residual_unit_3/batch_normalization_7/ReadVariableOp_1�Kmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_3/batch_normalization_8/ReadVariableOp�<model/residual_unit_3/batch_normalization_8/ReadVariableOp_1�Kmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Mmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�:model/residual_unit_3/batch_normalization_9/ReadVariableOp�<model/residual_unit_3/batch_normalization_9/ReadVariableOp_1�5model/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp�4model/residual_unit_3/conv2d_8/Conv2D/ReadVariableOp�4model/residual_unit_3/conv2d_9/Conv2D/ReadVariableOp�Lmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp�Nmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�;model/residual_unit_4/batch_normalization_10/ReadVariableOp�=model/residual_unit_4/batch_normalization_10/ReadVariableOp_1�Lmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp�Nmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�;model/residual_unit_4/batch_normalization_11/ReadVariableOp�=model/residual_unit_4/batch_normalization_11/ReadVariableOp_1�5model/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp�5model/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp�Lmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp�Nmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�;model/residual_unit_5/batch_normalization_12/ReadVariableOp�=model/residual_unit_5/batch_normalization_12/ReadVariableOp_1�Lmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp�Nmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�;model/residual_unit_5/batch_normalization_13/ReadVariableOp�=model/residual_unit_5/batch_normalization_13/ReadVariableOp_1�Lmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp�Nmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�;model/residual_unit_5/batch_normalization_14/ReadVariableOp�=model/residual_unit_5/batch_normalization_14/ReadVariableOp_1�5model/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp�5model/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp�5model/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp�
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
u
model/activation/ReluRelumodel/conv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������@@@�
model/max_pooling2d/MaxPoolMaxPool#model/activation/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingSAME*
strides
�
2model/residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;model_residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
#model/residual_unit/conv2d_1/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0:model/residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
6model/residual_unit/batch_normalization/ReadVariableOpReadVariableOp?model_residual_unit_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
8model/residual_unit/batch_normalization/ReadVariableOp_1ReadVariableOpAmodel_residual_unit_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Gmodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_residual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Imodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_residual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
8model/residual_unit/batch_normalization/FusedBatchNormV3FusedBatchNormV3,model/residual_unit/conv2d_1/Conv2D:output:0>model/residual_unit/batch_normalization/ReadVariableOp:value:0@model/residual_unit/batch_normalization/ReadVariableOp_1:value:0Omodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( �
model/residual_unit/ReluRelu<model/residual_unit/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
2model/residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;model_residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
#model/residual_unit/conv2d_2/Conv2DConv2D&model/residual_unit/Relu:activations:0:model/residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
8model/residual_unit/batch_normalization_1/ReadVariableOpReadVariableOpAmodel_residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
:model/residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOpCmodel_residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Imodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Kmodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_residual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
:model/residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3,model/residual_unit/conv2d_2/Conv2D:output:0@model/residual_unit/batch_normalization_1/ReadVariableOp:value:0Bmodel/residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Qmodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Smodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( �
model/residual_unit/addAddV2>model/residual_unit/batch_normalization_1/FusedBatchNormV3:y:0$model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:���������  @y
model/residual_unit/Relu_1Relumodel/residual_unit/add:z:0*
T0*/
_output_shapes
:���������  @�
4model/residual_unit_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_1_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
%model/residual_unit_1/conv2d_3/Conv2DConv2D(model/residual_unit/Relu_1:activations:0<model/residual_unit_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_1/batch_normalization_2/ReadVariableOpReadVariableOpCmodel_residual_unit_1_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_1/batch_normalization_2/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_1/conv2d_3/Conv2D:output:0Bmodel/residual_unit_1/batch_normalization_2/ReadVariableOp:value:0Dmodel/residual_unit_1/batch_normalization_2/ReadVariableOp_1:value:0Smodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_1/ReluRelu@model/residual_unit_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
4model/residual_unit_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_1_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%model/residual_unit_1/conv2d_4/Conv2DConv2D(model/residual_unit_1/Relu:activations:0<model/residual_unit_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_1/batch_normalization_3/ReadVariableOpReadVariableOpCmodel_residual_unit_1_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_1/batch_normalization_3/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_1/conv2d_4/Conv2D:output:0Bmodel/residual_unit_1/batch_normalization_3/ReadVariableOp:value:0Dmodel/residual_unit_1/batch_normalization_3/ReadVariableOp_1:value:0Smodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
4model/residual_unit_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_1_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
%model/residual_unit_1/conv2d_5/Conv2DConv2D(model/residual_unit/Relu_1:activations:0<model/residual_unit_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_1/batch_normalization_4/ReadVariableOpReadVariableOpCmodel_residual_unit_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_1/batch_normalization_4/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_1/conv2d_5/Conv2D:output:0Bmodel/residual_unit_1/batch_normalization_4/ReadVariableOp:value:0Dmodel/residual_unit_1/batch_normalization_4/ReadVariableOp_1:value:0Smodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_1/addAddV2@model/residual_unit_1/batch_normalization_3/FusedBatchNormV3:y:0@model/residual_unit_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~
model/residual_unit_1/Relu_1Relumodel/residual_unit_1/add:z:0*
T0*0
_output_shapes
:�����������
4model/residual_unit_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_2_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%model/residual_unit_2/conv2d_6/Conv2DConv2D*model/residual_unit_1/Relu_1:activations:0<model/residual_unit_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_2/batch_normalization_5/ReadVariableOpReadVariableOpCmodel_residual_unit_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_2/batch_normalization_5/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_2/conv2d_6/Conv2D:output:0Bmodel/residual_unit_2/batch_normalization_5/ReadVariableOp:value:0Dmodel/residual_unit_2/batch_normalization_5/ReadVariableOp_1:value:0Smodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_2/ReluRelu@model/residual_unit_2/batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
4model/residual_unit_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_2_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%model/residual_unit_2/conv2d_7/Conv2DConv2D(model/residual_unit_2/Relu:activations:0<model/residual_unit_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_2/batch_normalization_6/ReadVariableOpReadVariableOpCmodel_residual_unit_2_batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_2/batch_normalization_6/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_2/conv2d_7/Conv2D:output:0Bmodel/residual_unit_2/batch_normalization_6/ReadVariableOp:value:0Dmodel/residual_unit_2/batch_normalization_6/ReadVariableOp_1:value:0Smodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_2/addAddV2@model/residual_unit_2/batch_normalization_6/FusedBatchNormV3:y:0*model/residual_unit_1/Relu_1:activations:0*
T0*0
_output_shapes
:����������~
model/residual_unit_2/Relu_1Relumodel/residual_unit_2/add:z:0*
T0*0
_output_shapes
:�����������
4model/residual_unit_3/conv2d_8/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_3_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%model/residual_unit_3/conv2d_8/Conv2DConv2D*model/residual_unit_2/Relu_1:activations:0<model/residual_unit_3/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_3/batch_normalization_7/ReadVariableOpReadVariableOpCmodel_residual_unit_3_batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_3/batch_normalization_7/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_3_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_3/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_3/conv2d_8/Conv2D:output:0Bmodel/residual_unit_3/batch_normalization_7/ReadVariableOp:value:0Dmodel/residual_unit_3/batch_normalization_7/ReadVariableOp_1:value:0Smodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_3/ReluRelu@model/residual_unit_3/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
4model/residual_unit_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp=model_residual_unit_3_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%model/residual_unit_3/conv2d_9/Conv2DConv2D(model/residual_unit_3/Relu:activations:0<model/residual_unit_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_3/batch_normalization_8/ReadVariableOpReadVariableOpCmodel_residual_unit_3_batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_3/batch_normalization_8/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_3_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_3/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3.model/residual_unit_3/conv2d_9/Conv2D:output:0Bmodel/residual_unit_3/batch_normalization_8/ReadVariableOp:value:0Dmodel/residual_unit_3/batch_normalization_8/ReadVariableOp_1:value:0Smodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
5model/residual_unit_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp>model_residual_unit_3_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&model/residual_unit_3/conv2d_10/Conv2DConv2D*model/residual_unit_2/Relu_1:activations:0=model/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
:model/residual_unit_3/batch_normalization_9/ReadVariableOpReadVariableOpCmodel_residual_unit_3_batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_3/batch_normalization_9/ReadVariableOp_1ReadVariableOpEmodel_residual_unit_3_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpTmodel_residual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVmodel_residual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<model/residual_unit_3/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3/model/residual_unit_3/conv2d_10/Conv2D:output:0Bmodel/residual_unit_3/batch_normalization_9/ReadVariableOp:value:0Dmodel/residual_unit_3/batch_normalization_9/ReadVariableOp_1:value:0Smodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Umodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_3/addAddV2@model/residual_unit_3/batch_normalization_8/FusedBatchNormV3:y:0@model/residual_unit_3/batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~
model/residual_unit_3/Relu_1Relumodel/residual_unit_3/add:z:0*
T0*0
_output_shapes
:�����������
5model/residual_unit_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp>model_residual_unit_4_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&model/residual_unit_4/conv2d_11/Conv2DConv2D*model/residual_unit_3/Relu_1:activations:0=model/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
;model/residual_unit_4/batch_normalization_10/ReadVariableOpReadVariableOpDmodel_residual_unit_4_batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_4/batch_normalization_10/ReadVariableOp_1ReadVariableOpFmodel_residual_unit_4_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Lmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_residual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Nmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_residual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_4/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3/model/residual_unit_4/conv2d_11/Conv2D:output:0Cmodel/residual_unit_4/batch_normalization_10/ReadVariableOp:value:0Emodel/residual_unit_4/batch_normalization_10/ReadVariableOp_1:value:0Tmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Vmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_4/ReluReluAmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
5model/residual_unit_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp>model_residual_unit_4_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&model/residual_unit_4/conv2d_12/Conv2DConv2D(model/residual_unit_4/Relu:activations:0=model/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
;model/residual_unit_4/batch_normalization_11/ReadVariableOpReadVariableOpDmodel_residual_unit_4_batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_4/batch_normalization_11/ReadVariableOp_1ReadVariableOpFmodel_residual_unit_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Lmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_residual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Nmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_residual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3/model/residual_unit_4/conv2d_12/Conv2D:output:0Cmodel/residual_unit_4/batch_normalization_11/ReadVariableOp:value:0Emodel/residual_unit_4/batch_normalization_11/ReadVariableOp_1:value:0Tmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Vmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_4/addAddV2Amodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3:y:0*model/residual_unit_3/Relu_1:activations:0*
T0*0
_output_shapes
:����������~
model/residual_unit_4/Relu_1Relumodel/residual_unit_4/add:z:0*
T0*0
_output_shapes
:�����������
5model/residual_unit_5/conv2d_13/Conv2D/ReadVariableOpReadVariableOp>model_residual_unit_5_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&model/residual_unit_5/conv2d_13/Conv2DConv2D*model/residual_unit_4/Relu_1:activations:0=model/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
;model/residual_unit_5/batch_normalization_12/ReadVariableOpReadVariableOpDmodel_residual_unit_5_batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_5/batch_normalization_12/ReadVariableOp_1ReadVariableOpFmodel_residual_unit_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Lmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_residual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Nmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_residual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3/model/residual_unit_5/conv2d_13/Conv2D:output:0Cmodel/residual_unit_5/batch_normalization_12/ReadVariableOp:value:0Emodel/residual_unit_5/batch_normalization_12/ReadVariableOp_1:value:0Tmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Vmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_5/ReluReluAmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
5model/residual_unit_5/conv2d_14/Conv2D/ReadVariableOpReadVariableOp>model_residual_unit_5_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&model/residual_unit_5/conv2d_14/Conv2DConv2D(model/residual_unit_5/Relu:activations:0=model/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
;model/residual_unit_5/batch_normalization_13/ReadVariableOpReadVariableOpDmodel_residual_unit_5_batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_5/batch_normalization_13/ReadVariableOp_1ReadVariableOpFmodel_residual_unit_5_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Lmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_residual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Nmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_residual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_5/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3/model/residual_unit_5/conv2d_14/Conv2D:output:0Cmodel/residual_unit_5/batch_normalization_13/ReadVariableOp:value:0Emodel/residual_unit_5/batch_normalization_13/ReadVariableOp_1:value:0Tmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Vmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
5model/residual_unit_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp>model_residual_unit_5_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&model/residual_unit_5/conv2d_15/Conv2DConv2D*model/residual_unit_4/Relu_1:activations:0=model/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
;model/residual_unit_5/batch_normalization_14/ReadVariableOpReadVariableOpDmodel_residual_unit_5_batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_5/batch_normalization_14/ReadVariableOp_1ReadVariableOpFmodel_residual_unit_5_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Lmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpUmodel_residual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Nmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWmodel_residual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model/residual_unit_5/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3/model/residual_unit_5/conv2d_15/Conv2D:output:0Cmodel/residual_unit_5/batch_normalization_14/ReadVariableOp:value:0Emodel/residual_unit_5/batch_normalization_14/ReadVariableOp_1:value:0Tmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Vmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model/residual_unit_5/addAddV2Amodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3:y:0Amodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������~
model/residual_unit_5/Relu_1Relumodel/residual_unit_5/add:z:0*
T0*0
_output_shapes
:�����������
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
#model/global_average_pooling2d/MeanMean*model/residual_unit_5/Relu_1:activations:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape,model/global_average_pooling2d/Mean:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
model/sampling/ShapeShapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:f
!model/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#model/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1model/sampling/random_normal/RandomStandardNormalRandomStandardNormalmodel/sampling/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0�
 model/sampling/random_normal/mulMul:model/sampling/random_normal/RandomStandardNormal:output:0,model/sampling/random_normal/stddev:output:0*
T0*(
_output_shapes
:�����������
model/sampling/random_normalAddV2$model/sampling/random_normal/mul:z:0*model/sampling/random_normal/mean:output:0*
T0*(
_output_shapes
:����������]
model/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/sampling/truedivRealDivmodel/dense_1/BiasAdd:output:0!model/sampling/truediv/y:output:0*
T0*(
_output_shapes
:����������h
model/sampling/ExpExpmodel/sampling/truediv:z:0*
T0*(
_output_shapes
:�����������
model/sampling/mulMul model/sampling/random_normal:z:0model/sampling/Exp:y:0*
T0*(
_output_shapes
:�����������
model/sampling/addAddV2model/sampling/mul:z:0model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������l
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������p

Identity_1Identitymodel/dense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������h

Identity_2Identitymodel/sampling/add:z:0^NoOp*
T0*(
_output_shapes
:�����������)
NoOpNoOp#^model/conv2d/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOpH^model/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpJ^model/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_17^model/residual_unit/batch_normalization/ReadVariableOp9^model/residual_unit/batch_normalization/ReadVariableOp_1J^model/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpL^model/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_19^model/residual_unit/batch_normalization_1/ReadVariableOp;^model/residual_unit/batch_normalization_1/ReadVariableOp_13^model/residual_unit/conv2d_1/Conv2D/ReadVariableOp3^model/residual_unit/conv2d_2/Conv2D/ReadVariableOpL^model/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_1/batch_normalization_2/ReadVariableOp=^model/residual_unit_1/batch_normalization_2/ReadVariableOp_1L^model/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_1/batch_normalization_3/ReadVariableOp=^model/residual_unit_1/batch_normalization_3/ReadVariableOp_1L^model/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_1/batch_normalization_4/ReadVariableOp=^model/residual_unit_1/batch_normalization_4/ReadVariableOp_15^model/residual_unit_1/conv2d_3/Conv2D/ReadVariableOp5^model/residual_unit_1/conv2d_4/Conv2D/ReadVariableOp5^model/residual_unit_1/conv2d_5/Conv2D/ReadVariableOpL^model/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_2/batch_normalization_5/ReadVariableOp=^model/residual_unit_2/batch_normalization_5/ReadVariableOp_1L^model/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_2/batch_normalization_6/ReadVariableOp=^model/residual_unit_2/batch_normalization_6/ReadVariableOp_15^model/residual_unit_2/conv2d_6/Conv2D/ReadVariableOp5^model/residual_unit_2/conv2d_7/Conv2D/ReadVariableOpL^model/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_3/batch_normalization_7/ReadVariableOp=^model/residual_unit_3/batch_normalization_7/ReadVariableOp_1L^model/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_3/batch_normalization_8/ReadVariableOp=^model/residual_unit_3/batch_normalization_8/ReadVariableOp_1L^model/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpN^model/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1;^model/residual_unit_3/batch_normalization_9/ReadVariableOp=^model/residual_unit_3/batch_normalization_9/ReadVariableOp_16^model/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp5^model/residual_unit_3/conv2d_8/Conv2D/ReadVariableOp5^model/residual_unit_3/conv2d_9/Conv2D/ReadVariableOpM^model/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpO^model/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1<^model/residual_unit_4/batch_normalization_10/ReadVariableOp>^model/residual_unit_4/batch_normalization_10/ReadVariableOp_1M^model/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpO^model/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1<^model/residual_unit_4/batch_normalization_11/ReadVariableOp>^model/residual_unit_4/batch_normalization_11/ReadVariableOp_16^model/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp6^model/residual_unit_4/conv2d_12/Conv2D/ReadVariableOpM^model/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpO^model/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1<^model/residual_unit_5/batch_normalization_12/ReadVariableOp>^model/residual_unit_5/batch_normalization_12/ReadVariableOp_1M^model/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpO^model/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1<^model/residual_unit_5/batch_normalization_13/ReadVariableOp>^model/residual_unit_5/batch_normalization_13/ReadVariableOp_1M^model/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpO^model/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1<^model/residual_unit_5/batch_normalization_14/ReadVariableOp>^model/residual_unit_5/batch_normalization_14/ReadVariableOp_16^model/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp6^model/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp6^model/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2�
Gmodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpGmodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
Imodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Imodel/residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_12p
6model/residual_unit/batch_normalization/ReadVariableOp6model/residual_unit/batch_normalization/ReadVariableOp2t
8model/residual_unit/batch_normalization/ReadVariableOp_18model/residual_unit/batch_normalization/ReadVariableOp_12�
Imodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpImodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Kmodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Kmodel/residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12t
8model/residual_unit/batch_normalization_1/ReadVariableOp8model/residual_unit/batch_normalization_1/ReadVariableOp2x
:model/residual_unit/batch_normalization_1/ReadVariableOp_1:model/residual_unit/batch_normalization_1/ReadVariableOp_12h
2model/residual_unit/conv2d_1/Conv2D/ReadVariableOp2model/residual_unit/conv2d_1/Conv2D/ReadVariableOp2h
2model/residual_unit/conv2d_2/Conv2D/ReadVariableOp2model/residual_unit/conv2d_2/Conv2D/ReadVariableOp2�
Kmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_1/batch_normalization_2/ReadVariableOp:model/residual_unit_1/batch_normalization_2/ReadVariableOp2|
<model/residual_unit_1/batch_normalization_2/ReadVariableOp_1<model/residual_unit_1/batch_normalization_2/ReadVariableOp_12�
Kmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_1/batch_normalization_3/ReadVariableOp:model/residual_unit_1/batch_normalization_3/ReadVariableOp2|
<model/residual_unit_1/batch_normalization_3/ReadVariableOp_1<model/residual_unit_1/batch_normalization_3/ReadVariableOp_12�
Kmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_1/batch_normalization_4/ReadVariableOp:model/residual_unit_1/batch_normalization_4/ReadVariableOp2|
<model/residual_unit_1/batch_normalization_4/ReadVariableOp_1<model/residual_unit_1/batch_normalization_4/ReadVariableOp_12l
4model/residual_unit_1/conv2d_3/Conv2D/ReadVariableOp4model/residual_unit_1/conv2d_3/Conv2D/ReadVariableOp2l
4model/residual_unit_1/conv2d_4/Conv2D/ReadVariableOp4model/residual_unit_1/conv2d_4/Conv2D/ReadVariableOp2l
4model/residual_unit_1/conv2d_5/Conv2D/ReadVariableOp4model/residual_unit_1/conv2d_5/Conv2D/ReadVariableOp2�
Kmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_2/batch_normalization_5/ReadVariableOp:model/residual_unit_2/batch_normalization_5/ReadVariableOp2|
<model/residual_unit_2/batch_normalization_5/ReadVariableOp_1<model/residual_unit_2/batch_normalization_5/ReadVariableOp_12�
Kmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_2/batch_normalization_6/ReadVariableOp:model/residual_unit_2/batch_normalization_6/ReadVariableOp2|
<model/residual_unit_2/batch_normalization_6/ReadVariableOp_1<model/residual_unit_2/batch_normalization_6/ReadVariableOp_12l
4model/residual_unit_2/conv2d_6/Conv2D/ReadVariableOp4model/residual_unit_2/conv2d_6/Conv2D/ReadVariableOp2l
4model/residual_unit_2/conv2d_7/Conv2D/ReadVariableOp4model/residual_unit_2/conv2d_7/Conv2D/ReadVariableOp2�
Kmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_3/batch_normalization_7/ReadVariableOp:model/residual_unit_3/batch_normalization_7/ReadVariableOp2|
<model/residual_unit_3/batch_normalization_7/ReadVariableOp_1<model/residual_unit_3/batch_normalization_7/ReadVariableOp_12�
Kmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_3/batch_normalization_8/ReadVariableOp:model/residual_unit_3/batch_normalization_8/ReadVariableOp2|
<model/residual_unit_3/batch_normalization_8/ReadVariableOp_1<model/residual_unit_3/batch_normalization_8/ReadVariableOp_12�
Kmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpKmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Mmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Mmodel/residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12x
:model/residual_unit_3/batch_normalization_9/ReadVariableOp:model/residual_unit_3/batch_normalization_9/ReadVariableOp2|
<model/residual_unit_3/batch_normalization_9/ReadVariableOp_1<model/residual_unit_3/batch_normalization_9/ReadVariableOp_12n
5model/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp5model/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp2l
4model/residual_unit_3/conv2d_8/Conv2D/ReadVariableOp4model/residual_unit_3/conv2d_8/Conv2D/ReadVariableOp2l
4model/residual_unit_3/conv2d_9/Conv2D/ReadVariableOp4model/residual_unit_3/conv2d_9/Conv2D/ReadVariableOp2�
Lmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpLmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2�
Nmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Nmodel/residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12z
;model/residual_unit_4/batch_normalization_10/ReadVariableOp;model/residual_unit_4/batch_normalization_10/ReadVariableOp2~
=model/residual_unit_4/batch_normalization_10/ReadVariableOp_1=model/residual_unit_4/batch_normalization_10/ReadVariableOp_12�
Lmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpLmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2�
Nmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Nmodel/residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12z
;model/residual_unit_4/batch_normalization_11/ReadVariableOp;model/residual_unit_4/batch_normalization_11/ReadVariableOp2~
=model/residual_unit_4/batch_normalization_11/ReadVariableOp_1=model/residual_unit_4/batch_normalization_11/ReadVariableOp_12n
5model/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp5model/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp2n
5model/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp5model/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp2�
Lmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpLmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2�
Nmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Nmodel/residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12z
;model/residual_unit_5/batch_normalization_12/ReadVariableOp;model/residual_unit_5/batch_normalization_12/ReadVariableOp2~
=model/residual_unit_5/batch_normalization_12/ReadVariableOp_1=model/residual_unit_5/batch_normalization_12/ReadVariableOp_12�
Lmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpLmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2�
Nmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Nmodel/residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12z
;model/residual_unit_5/batch_normalization_13/ReadVariableOp;model/residual_unit_5/batch_normalization_13/ReadVariableOp2~
=model/residual_unit_5/batch_normalization_13/ReadVariableOp_1=model/residual_unit_5/batch_normalization_13/ReadVariableOp_12�
Lmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpLmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2�
Nmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Nmodel/residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12z
;model/residual_unit_5/batch_normalization_14/ReadVariableOp;model/residual_unit_5/batch_normalization_14/ReadVariableOp2~
=model/residual_unit_5/batch_normalization_14/ReadVariableOp_1=model/residual_unit_5/batch_normalization_14/ReadVariableOp_12n
5model/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp5model/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp2n
5model/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp5model/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp2n
5model/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp5model/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023334

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_layer_call_and_return_conditional_losses_128018056

inputs8
conv2d_readvariableop_resource:@
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������@@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�0
�	
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128018262

inputsC
'conv2d_6_conv2d_readvariableop_resource:��<
-batch_normalization_5_readvariableop_resource:	�>
/batch_normalization_5_readvariableop_1_resource:	�M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_7_conv2d_readvariableop_resource:��<
-batch_normalization_6_readvariableop_resource:	�>
/batch_normalization_6_readvariableop_1_resource:	�M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( s
ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2DRelu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( {
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1^conv2d_6/Conv2D/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_14_layer_call_fn_128023608

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128018018�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_7_layer_call_fn_128023161

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128017539�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128017155

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
��
�/
"__inference__traced_save_128023909
file_prefix,
(savev2_conv2d_kernel_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop<
8savev2_residual_unit_conv2d_1_kernel_read_readvariableopF
Bsavev2_residual_unit_batch_normalization_gamma_read_readvariableopE
Asavev2_residual_unit_batch_normalization_beta_read_readvariableop<
8savev2_residual_unit_conv2d_2_kernel_read_readvariableopH
Dsavev2_residual_unit_batch_normalization_1_gamma_read_readvariableopG
Csavev2_residual_unit_batch_normalization_1_beta_read_readvariableopL
Hsavev2_residual_unit_batch_normalization_moving_mean_read_readvariableopP
Lsavev2_residual_unit_batch_normalization_moving_variance_read_readvariableopN
Jsavev2_residual_unit_batch_normalization_1_moving_mean_read_readvariableopR
Nsavev2_residual_unit_batch_normalization_1_moving_variance_read_readvariableop>
:savev2_residual_unit_1_conv2d_3_kernel_read_readvariableopJ
Fsavev2_residual_unit_1_batch_normalization_2_gamma_read_readvariableopI
Esavev2_residual_unit_1_batch_normalization_2_beta_read_readvariableop>
:savev2_residual_unit_1_conv2d_4_kernel_read_readvariableopJ
Fsavev2_residual_unit_1_batch_normalization_3_gamma_read_readvariableopI
Esavev2_residual_unit_1_batch_normalization_3_beta_read_readvariableop>
:savev2_residual_unit_1_conv2d_5_kernel_read_readvariableopJ
Fsavev2_residual_unit_1_batch_normalization_4_gamma_read_readvariableopI
Esavev2_residual_unit_1_batch_normalization_4_beta_read_readvariableopP
Lsavev2_residual_unit_1_batch_normalization_2_moving_mean_read_readvariableopT
Psavev2_residual_unit_1_batch_normalization_2_moving_variance_read_readvariableopP
Lsavev2_residual_unit_1_batch_normalization_3_moving_mean_read_readvariableopT
Psavev2_residual_unit_1_batch_normalization_3_moving_variance_read_readvariableopP
Lsavev2_residual_unit_1_batch_normalization_4_moving_mean_read_readvariableopT
Psavev2_residual_unit_1_batch_normalization_4_moving_variance_read_readvariableop>
:savev2_residual_unit_2_conv2d_6_kernel_read_readvariableopJ
Fsavev2_residual_unit_2_batch_normalization_5_gamma_read_readvariableopI
Esavev2_residual_unit_2_batch_normalization_5_beta_read_readvariableop>
:savev2_residual_unit_2_conv2d_7_kernel_read_readvariableopJ
Fsavev2_residual_unit_2_batch_normalization_6_gamma_read_readvariableopI
Esavev2_residual_unit_2_batch_normalization_6_beta_read_readvariableopP
Lsavev2_residual_unit_2_batch_normalization_5_moving_mean_read_readvariableopT
Psavev2_residual_unit_2_batch_normalization_5_moving_variance_read_readvariableopP
Lsavev2_residual_unit_2_batch_normalization_6_moving_mean_read_readvariableopT
Psavev2_residual_unit_2_batch_normalization_6_moving_variance_read_readvariableop>
:savev2_residual_unit_3_conv2d_8_kernel_read_readvariableopJ
Fsavev2_residual_unit_3_batch_normalization_7_gamma_read_readvariableopI
Esavev2_residual_unit_3_batch_normalization_7_beta_read_readvariableop>
:savev2_residual_unit_3_conv2d_9_kernel_read_readvariableopJ
Fsavev2_residual_unit_3_batch_normalization_8_gamma_read_readvariableopI
Esavev2_residual_unit_3_batch_normalization_8_beta_read_readvariableop?
;savev2_residual_unit_3_conv2d_10_kernel_read_readvariableopJ
Fsavev2_residual_unit_3_batch_normalization_9_gamma_read_readvariableopI
Esavev2_residual_unit_3_batch_normalization_9_beta_read_readvariableopP
Lsavev2_residual_unit_3_batch_normalization_7_moving_mean_read_readvariableopT
Psavev2_residual_unit_3_batch_normalization_7_moving_variance_read_readvariableopP
Lsavev2_residual_unit_3_batch_normalization_8_moving_mean_read_readvariableopT
Psavev2_residual_unit_3_batch_normalization_8_moving_variance_read_readvariableopP
Lsavev2_residual_unit_3_batch_normalization_9_moving_mean_read_readvariableopT
Psavev2_residual_unit_3_batch_normalization_9_moving_variance_read_readvariableop?
;savev2_residual_unit_4_conv2d_11_kernel_read_readvariableopK
Gsavev2_residual_unit_4_batch_normalization_10_gamma_read_readvariableopJ
Fsavev2_residual_unit_4_batch_normalization_10_beta_read_readvariableop?
;savev2_residual_unit_4_conv2d_12_kernel_read_readvariableopK
Gsavev2_residual_unit_4_batch_normalization_11_gamma_read_readvariableopJ
Fsavev2_residual_unit_4_batch_normalization_11_beta_read_readvariableopQ
Msavev2_residual_unit_4_batch_normalization_10_moving_mean_read_readvariableopU
Qsavev2_residual_unit_4_batch_normalization_10_moving_variance_read_readvariableopQ
Msavev2_residual_unit_4_batch_normalization_11_moving_mean_read_readvariableopU
Qsavev2_residual_unit_4_batch_normalization_11_moving_variance_read_readvariableop?
;savev2_residual_unit_5_conv2d_13_kernel_read_readvariableopK
Gsavev2_residual_unit_5_batch_normalization_12_gamma_read_readvariableopJ
Fsavev2_residual_unit_5_batch_normalization_12_beta_read_readvariableop?
;savev2_residual_unit_5_conv2d_14_kernel_read_readvariableopK
Gsavev2_residual_unit_5_batch_normalization_13_gamma_read_readvariableopJ
Fsavev2_residual_unit_5_batch_normalization_13_beta_read_readvariableop?
;savev2_residual_unit_5_conv2d_15_kernel_read_readvariableopK
Gsavev2_residual_unit_5_batch_normalization_14_gamma_read_readvariableopJ
Fsavev2_residual_unit_5_batch_normalization_14_beta_read_readvariableopQ
Msavev2_residual_unit_5_batch_normalization_12_moving_mean_read_readvariableopU
Qsavev2_residual_unit_5_batch_normalization_12_moving_variance_read_readvariableopQ
Msavev2_residual_unit_5_batch_normalization_13_moving_mean_read_readvariableopU
Qsavev2_residual_unit_5_batch_normalization_13_moving_variance_read_readvariableopQ
Msavev2_residual_unit_5_batch_normalization_14_moving_mean_read_readvariableopU
Qsavev2_residual_unit_5_batch_normalization_14_moving_variance_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�
value�B�QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*�
value�B�QB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop8savev2_residual_unit_conv2d_1_kernel_read_readvariableopBsavev2_residual_unit_batch_normalization_gamma_read_readvariableopAsavev2_residual_unit_batch_normalization_beta_read_readvariableop8savev2_residual_unit_conv2d_2_kernel_read_readvariableopDsavev2_residual_unit_batch_normalization_1_gamma_read_readvariableopCsavev2_residual_unit_batch_normalization_1_beta_read_readvariableopHsavev2_residual_unit_batch_normalization_moving_mean_read_readvariableopLsavev2_residual_unit_batch_normalization_moving_variance_read_readvariableopJsavev2_residual_unit_batch_normalization_1_moving_mean_read_readvariableopNsavev2_residual_unit_batch_normalization_1_moving_variance_read_readvariableop:savev2_residual_unit_1_conv2d_3_kernel_read_readvariableopFsavev2_residual_unit_1_batch_normalization_2_gamma_read_readvariableopEsavev2_residual_unit_1_batch_normalization_2_beta_read_readvariableop:savev2_residual_unit_1_conv2d_4_kernel_read_readvariableopFsavev2_residual_unit_1_batch_normalization_3_gamma_read_readvariableopEsavev2_residual_unit_1_batch_normalization_3_beta_read_readvariableop:savev2_residual_unit_1_conv2d_5_kernel_read_readvariableopFsavev2_residual_unit_1_batch_normalization_4_gamma_read_readvariableopEsavev2_residual_unit_1_batch_normalization_4_beta_read_readvariableopLsavev2_residual_unit_1_batch_normalization_2_moving_mean_read_readvariableopPsavev2_residual_unit_1_batch_normalization_2_moving_variance_read_readvariableopLsavev2_residual_unit_1_batch_normalization_3_moving_mean_read_readvariableopPsavev2_residual_unit_1_batch_normalization_3_moving_variance_read_readvariableopLsavev2_residual_unit_1_batch_normalization_4_moving_mean_read_readvariableopPsavev2_residual_unit_1_batch_normalization_4_moving_variance_read_readvariableop:savev2_residual_unit_2_conv2d_6_kernel_read_readvariableopFsavev2_residual_unit_2_batch_normalization_5_gamma_read_readvariableopEsavev2_residual_unit_2_batch_normalization_5_beta_read_readvariableop:savev2_residual_unit_2_conv2d_7_kernel_read_readvariableopFsavev2_residual_unit_2_batch_normalization_6_gamma_read_readvariableopEsavev2_residual_unit_2_batch_normalization_6_beta_read_readvariableopLsavev2_residual_unit_2_batch_normalization_5_moving_mean_read_readvariableopPsavev2_residual_unit_2_batch_normalization_5_moving_variance_read_readvariableopLsavev2_residual_unit_2_batch_normalization_6_moving_mean_read_readvariableopPsavev2_residual_unit_2_batch_normalization_6_moving_variance_read_readvariableop:savev2_residual_unit_3_conv2d_8_kernel_read_readvariableopFsavev2_residual_unit_3_batch_normalization_7_gamma_read_readvariableopEsavev2_residual_unit_3_batch_normalization_7_beta_read_readvariableop:savev2_residual_unit_3_conv2d_9_kernel_read_readvariableopFsavev2_residual_unit_3_batch_normalization_8_gamma_read_readvariableopEsavev2_residual_unit_3_batch_normalization_8_beta_read_readvariableop;savev2_residual_unit_3_conv2d_10_kernel_read_readvariableopFsavev2_residual_unit_3_batch_normalization_9_gamma_read_readvariableopEsavev2_residual_unit_3_batch_normalization_9_beta_read_readvariableopLsavev2_residual_unit_3_batch_normalization_7_moving_mean_read_readvariableopPsavev2_residual_unit_3_batch_normalization_7_moving_variance_read_readvariableopLsavev2_residual_unit_3_batch_normalization_8_moving_mean_read_readvariableopPsavev2_residual_unit_3_batch_normalization_8_moving_variance_read_readvariableopLsavev2_residual_unit_3_batch_normalization_9_moving_mean_read_readvariableopPsavev2_residual_unit_3_batch_normalization_9_moving_variance_read_readvariableop;savev2_residual_unit_4_conv2d_11_kernel_read_readvariableopGsavev2_residual_unit_4_batch_normalization_10_gamma_read_readvariableopFsavev2_residual_unit_4_batch_normalization_10_beta_read_readvariableop;savev2_residual_unit_4_conv2d_12_kernel_read_readvariableopGsavev2_residual_unit_4_batch_normalization_11_gamma_read_readvariableopFsavev2_residual_unit_4_batch_normalization_11_beta_read_readvariableopMsavev2_residual_unit_4_batch_normalization_10_moving_mean_read_readvariableopQsavev2_residual_unit_4_batch_normalization_10_moving_variance_read_readvariableopMsavev2_residual_unit_4_batch_normalization_11_moving_mean_read_readvariableopQsavev2_residual_unit_4_batch_normalization_11_moving_variance_read_readvariableop;savev2_residual_unit_5_conv2d_13_kernel_read_readvariableopGsavev2_residual_unit_5_batch_normalization_12_gamma_read_readvariableopFsavev2_residual_unit_5_batch_normalization_12_beta_read_readvariableop;savev2_residual_unit_5_conv2d_14_kernel_read_readvariableopGsavev2_residual_unit_5_batch_normalization_13_gamma_read_readvariableopFsavev2_residual_unit_5_batch_normalization_13_beta_read_readvariableop;savev2_residual_unit_5_conv2d_15_kernel_read_readvariableopGsavev2_residual_unit_5_batch_normalization_14_gamma_read_readvariableopFsavev2_residual_unit_5_batch_normalization_14_beta_read_readvariableopMsavev2_residual_unit_5_batch_normalization_12_moving_mean_read_readvariableopQsavev2_residual_unit_5_batch_normalization_12_moving_variance_read_readvariableopMsavev2_residual_unit_5_batch_normalization_13_moving_mean_read_readvariableopQsavev2_residual_unit_5_batch_normalization_13_moving_variance_read_readvariableopMsavev2_residual_unit_5_batch_normalization_14_moving_mean_read_readvariableopQsavev2_residual_unit_5_batch_normalization_14_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:
��:�:
��:�:@@:@:@:@@:@:@:@:@:@:@:@�:�:�:��:�:�:@�:�:�:�:�:�:�:�:�:��:�:�:��:�:�:�:�:�:�:��:�:�:��:�:�:��:�:�:�:�:�:�:�:�:��:�:�:��:�:�:�:�:�:�:��:�:�:��:�:�:��:�:�:�:�:�:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:! 

_output_shapes	
:�:!!

_output_shapes	
:�:."*
(
_output_shapes
:��:!#

_output_shapes	
:�:!$

_output_shapes	
:�:!%

_output_shapes	
:�:!&

_output_shapes	
:�:!'

_output_shapes	
:�:!(

_output_shapes	
:�:.)*
(
_output_shapes
:��:!*

_output_shapes	
:�:!+

_output_shapes	
:�:.,*
(
_output_shapes
:��:!-

_output_shapes	
:�:!.

_output_shapes	
:�:./*
(
_output_shapes
:��:!0

_output_shapes	
:�:!1

_output_shapes	
:�:!2

_output_shapes	
:�:!3

_output_shapes	
:�:!4

_output_shapes	
:�:!5

_output_shapes	
:�:!6

_output_shapes	
:�:!7

_output_shapes	
:�:.8*
(
_output_shapes
:��:!9

_output_shapes	
:�:!:

_output_shapes	
:�:.;*
(
_output_shapes
:��:!<

_output_shapes	
:�:!=

_output_shapes	
:�:!>

_output_shapes	
:�:!?

_output_shapes	
:�:!@

_output_shapes	
:�:!A

_output_shapes	
:�:.B*
(
_output_shapes
:��:!C

_output_shapes	
:�:!D

_output_shapes	
:�:.E*
(
_output_shapes
:��:!F

_output_shapes	
:�:!G

_output_shapes	
:�:.H*
(
_output_shapes
:��:!I

_output_shapes	
:�:!J

_output_shapes	
:�:!K

_output_shapes	
:�:!L

_output_shapes	
:�:!M

_output_shapes	
:�:!N

_output_shapes	
:�:!O

_output_shapes	
:�:!P

_output_shapes	
:�:Q

_output_shapes
: 
�1
�

N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018415

inputsD
(conv2d_11_conv2d_readvariableop_resource:��=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_12_conv2d_readvariableop_resource:��=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�conv2d_11/Conv2D/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_11/Conv2D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( t
ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_12/Conv2DConv2DRelu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_12/Conv2D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( |
addAddV2+batch_normalization_11/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1 ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128017731

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
u
,__inference_sampling_layer_call_fn_128022698
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_128018584p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�F
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021938

inputsB
'conv2d_3_conv2d_readvariableop_resource:@�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��<
-batch_normalization_3_readvariableop_resource:	�>
/batch_normalization_3_readvariableop_1_resource:	�M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�B
'conv2d_5_conv2d_readvariableop_resource:@�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( s
ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2DRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������  @: : : : : : : : : : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022838

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
*__inference_conv2d_layer_call_fn_128021651

inputs!
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_128018056w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�[
D__inference_model_layer_call_and_return_conditional_losses_128021333

inputs?
%conv2d_conv2d_readvariableop_resource:@O
5residual_unit_conv2d_1_conv2d_readvariableop_resource:@@G
9residual_unit_batch_normalization_readvariableop_resource:@I
;residual_unit_batch_normalization_readvariableop_1_resource:@X
Jresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource:@Z
Lresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@O
5residual_unit_conv2d_2_conv2d_readvariableop_resource:@@I
;residual_unit_batch_normalization_1_readvariableop_resource:@K
=residual_unit_batch_normalization_1_readvariableop_1_resource:@Z
Lresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@\
Nresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@R
7residual_unit_1_conv2d_3_conv2d_readvariableop_resource:@�L
=residual_unit_1_batch_normalization_2_readvariableop_resource:	�N
?residual_unit_1_batch_normalization_2_readvariableop_1_resource:	�]
Nresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_1_conv2d_4_conv2d_readvariableop_resource:��L
=residual_unit_1_batch_normalization_3_readvariableop_resource:	�N
?residual_unit_1_batch_normalization_3_readvariableop_1_resource:	�]
Nresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�R
7residual_unit_1_conv2d_5_conv2d_readvariableop_resource:@�L
=residual_unit_1_batch_normalization_4_readvariableop_resource:	�N
?residual_unit_1_batch_normalization_4_readvariableop_1_resource:	�]
Nresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_2_conv2d_6_conv2d_readvariableop_resource:��L
=residual_unit_2_batch_normalization_5_readvariableop_resource:	�N
?residual_unit_2_batch_normalization_5_readvariableop_1_resource:	�]
Nresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_2_conv2d_7_conv2d_readvariableop_resource:��L
=residual_unit_2_batch_normalization_6_readvariableop_resource:	�N
?residual_unit_2_batch_normalization_6_readvariableop_1_resource:	�]
Nresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_3_conv2d_8_conv2d_readvariableop_resource:��L
=residual_unit_3_batch_normalization_7_readvariableop_resource:	�N
?residual_unit_3_batch_normalization_7_readvariableop_1_resource:	�]
Nresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�S
7residual_unit_3_conv2d_9_conv2d_readvariableop_resource:��L
=residual_unit_3_batch_normalization_8_readvariableop_resource:	�N
?residual_unit_3_batch_normalization_8_readvariableop_1_resource:	�]
Nresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_3_conv2d_10_conv2d_readvariableop_resource:��L
=residual_unit_3_batch_normalization_9_readvariableop_resource:	�N
?residual_unit_3_batch_normalization_9_readvariableop_1_resource:	�]
Nresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�_
Presidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_4_conv2d_11_conv2d_readvariableop_resource:��M
>residual_unit_4_batch_normalization_10_readvariableop_resource:	�O
@residual_unit_4_batch_normalization_10_readvariableop_1_resource:	�^
Oresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_4_conv2d_12_conv2d_readvariableop_resource:��M
>residual_unit_4_batch_normalization_11_readvariableop_resource:	�O
@residual_unit_4_batch_normalization_11_readvariableop_1_resource:	�^
Oresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_5_conv2d_13_conv2d_readvariableop_resource:��M
>residual_unit_5_batch_normalization_12_readvariableop_resource:	�O
@residual_unit_5_batch_normalization_12_readvariableop_1_resource:	�^
Oresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_5_conv2d_14_conv2d_readvariableop_resource:��M
>residual_unit_5_batch_normalization_13_readvariableop_resource:	�O
@residual_unit_5_batch_normalization_13_readvariableop_1_resource:	�^
Oresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�T
8residual_unit_5_conv2d_15_conv2d_readvariableop_resource:��M
>residual_unit_5_batch_normalization_14_readvariableop_resource:	�O
@residual_unit_5_batch_normalization_14_readvariableop_1_resource:	�^
Oresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�`
Qresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��conv2d/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�Aresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp�Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�0residual_unit/batch_normalization/ReadVariableOp�2residual_unit/batch_normalization/ReadVariableOp_1�Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�2residual_unit/batch_normalization_1/ReadVariableOp�4residual_unit/batch_normalization_1/ReadVariableOp_1�,residual_unit/conv2d_1/Conv2D/ReadVariableOp�,residual_unit/conv2d_2/Conv2D/ReadVariableOp�Eresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_1/batch_normalization_2/ReadVariableOp�6residual_unit_1/batch_normalization_2/ReadVariableOp_1�Eresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_1/batch_normalization_3/ReadVariableOp�6residual_unit_1/batch_normalization_3/ReadVariableOp_1�Eresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_1/batch_normalization_4/ReadVariableOp�6residual_unit_1/batch_normalization_4/ReadVariableOp_1�.residual_unit_1/conv2d_3/Conv2D/ReadVariableOp�.residual_unit_1/conv2d_4/Conv2D/ReadVariableOp�.residual_unit_1/conv2d_5/Conv2D/ReadVariableOp�Eresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_2/batch_normalization_5/ReadVariableOp�6residual_unit_2/batch_normalization_5/ReadVariableOp_1�Eresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_2/batch_normalization_6/ReadVariableOp�6residual_unit_2/batch_normalization_6/ReadVariableOp_1�.residual_unit_2/conv2d_6/Conv2D/ReadVariableOp�.residual_unit_2/conv2d_7/Conv2D/ReadVariableOp�Eresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_3/batch_normalization_7/ReadVariableOp�6residual_unit_3/batch_normalization_7/ReadVariableOp_1�Eresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_3/batch_normalization_8/ReadVariableOp�6residual_unit_3/batch_normalization_8/ReadVariableOp_1�Eresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�4residual_unit_3/batch_normalization_9/ReadVariableOp�6residual_unit_3/batch_normalization_9/ReadVariableOp_1�/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp�.residual_unit_3/conv2d_8/Conv2D/ReadVariableOp�.residual_unit_3/conv2d_9/Conv2D/ReadVariableOp�Fresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_4/batch_normalization_10/ReadVariableOp�7residual_unit_4/batch_normalization_10/ReadVariableOp_1�Fresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_4/batch_normalization_11/ReadVariableOp�7residual_unit_4/batch_normalization_11/ReadVariableOp_1�/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp�/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp�Fresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_5/batch_normalization_12/ReadVariableOp�7residual_unit_5/batch_normalization_12/ReadVariableOp_1�Fresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_5/batch_normalization_13/ReadVariableOp�7residual_unit_5/batch_normalization_13/ReadVariableOp_1�Fresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp�Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�5residual_unit_5/batch_normalization_14/ReadVariableOp�7residual_unit_5/batch_normalization_14/ReadVariableOp_1�/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp�/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp�/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
i
activation/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������@@@�
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:���������  @*
ksize
*
paddingSAME*
strides
�
,residual_unit/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
residual_unit/conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:04residual_unit/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
0residual_unit/batch_normalization/ReadVariableOpReadVariableOp9residual_unit_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
2residual_unit/batch_normalization/ReadVariableOp_1ReadVariableOp;residual_unit_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Aresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpJresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLresidual_unit_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
2residual_unit/batch_normalization/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_1/Conv2D:output:08residual_unit/batch_normalization/ReadVariableOp:value:0:residual_unit/batch_normalization/ReadVariableOp_1:value:0Iresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Kresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( �
residual_unit/ReluRelu6residual_unit/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
,residual_unit/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5residual_unit_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
residual_unit/conv2d_2/Conv2DConv2D residual_unit/Relu:activations:04residual_unit/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
2residual_unit/batch_normalization_1/ReadVariableOpReadVariableOp;residual_unit_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
4residual_unit/batch_normalization_1/ReadVariableOp_1ReadVariableOp=residual_unit_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpLresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNresidual_unit_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
4residual_unit/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&residual_unit/conv2d_2/Conv2D:output:0:residual_unit/batch_normalization_1/ReadVariableOp:value:0<residual_unit/batch_normalization_1/ReadVariableOp_1:value:0Kresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Mresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( �
residual_unit/addAddV28residual_unit/batch_normalization_1/FusedBatchNormV3:y:0max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:���������  @m
residual_unit/Relu_1Reluresidual_unit/add:z:0*
T0*/
_output_shapes
:���������  @�
.residual_unit_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp7residual_unit_1_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
residual_unit_1/conv2d_3/Conv2DConv2D"residual_unit/Relu_1:activations:06residual_unit_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_1/batch_normalization_2/ReadVariableOpReadVariableOp=residual_unit_1_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp?residual_unit_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3(residual_unit_1/conv2d_3/Conv2D:output:0<residual_unit_1/batch_normalization_2/ReadVariableOp:value:0>residual_unit_1/batch_normalization_2/ReadVariableOp_1:value:0Mresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_1/ReluRelu:residual_unit_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
.residual_unit_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp7residual_unit_1_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_1/conv2d_4/Conv2DConv2D"residual_unit_1/Relu:activations:06residual_unit_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_1/batch_normalization_3/ReadVariableOpReadVariableOp=residual_unit_1_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp?residual_unit_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3(residual_unit_1/conv2d_4/Conv2D:output:0<residual_unit_1/batch_normalization_3/ReadVariableOp:value:0>residual_unit_1/batch_normalization_3/ReadVariableOp_1:value:0Mresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
.residual_unit_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7residual_unit_1_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
residual_unit_1/conv2d_5/Conv2DConv2D"residual_unit/Relu_1:activations:06residual_unit_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_1/batch_normalization_4/ReadVariableOpReadVariableOp=residual_unit_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp?residual_unit_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3(residual_unit_1/conv2d_5/Conv2D:output:0<residual_unit_1/batch_normalization_4/ReadVariableOp:value:0>residual_unit_1/batch_normalization_4/ReadVariableOp_1:value:0Mresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_1/addAddV2:residual_unit_1/batch_normalization_3/FusedBatchNormV3:y:0:residual_unit_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������r
residual_unit_1/Relu_1Reluresidual_unit_1/add:z:0*
T0*0
_output_shapes
:�����������
.residual_unit_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp7residual_unit_2_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_2/conv2d_6/Conv2DConv2D$residual_unit_1/Relu_1:activations:06residual_unit_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_2/batch_normalization_5/ReadVariableOpReadVariableOp=residual_unit_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_5/ReadVariableOp_1ReadVariableOp?residual_unit_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3(residual_unit_2/conv2d_6/Conv2D:output:0<residual_unit_2/batch_normalization_5/ReadVariableOp:value:0>residual_unit_2/batch_normalization_5/ReadVariableOp_1:value:0Mresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_2/ReluRelu:residual_unit_2/batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
.residual_unit_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp7residual_unit_2_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_2/conv2d_7/Conv2DConv2D"residual_unit_2/Relu:activations:06residual_unit_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_2/batch_normalization_6/ReadVariableOpReadVariableOp=residual_unit_2_batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp?residual_unit_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3(residual_unit_2/conv2d_7/Conv2D:output:0<residual_unit_2/batch_normalization_6/ReadVariableOp:value:0>residual_unit_2/batch_normalization_6/ReadVariableOp_1:value:0Mresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_2/addAddV2:residual_unit_2/batch_normalization_6/FusedBatchNormV3:y:0$residual_unit_1/Relu_1:activations:0*
T0*0
_output_shapes
:����������r
residual_unit_2/Relu_1Reluresidual_unit_2/add:z:0*
T0*0
_output_shapes
:�����������
.residual_unit_3/conv2d_8/Conv2D/ReadVariableOpReadVariableOp7residual_unit_3_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_3/conv2d_8/Conv2DConv2D$residual_unit_2/Relu_1:activations:06residual_unit_3/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_3/batch_normalization_7/ReadVariableOpReadVariableOp=residual_unit_3_batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_7/ReadVariableOp_1ReadVariableOp?residual_unit_3_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_3_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(residual_unit_3/conv2d_8/Conv2D:output:0<residual_unit_3/batch_normalization_7/ReadVariableOp:value:0>residual_unit_3/batch_normalization_7/ReadVariableOp_1:value:0Mresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_3/ReluRelu:residual_unit_3/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
.residual_unit_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp7residual_unit_3_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
residual_unit_3/conv2d_9/Conv2DConv2D"residual_unit_3/Relu:activations:06residual_unit_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_3/batch_normalization_8/ReadVariableOpReadVariableOp=residual_unit_3_batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_8/ReadVariableOp_1ReadVariableOp?residual_unit_3_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_3_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3(residual_unit_3/conv2d_9/Conv2D:output:0<residual_unit_3/batch_normalization_8/ReadVariableOp:value:0>residual_unit_3/batch_normalization_8/ReadVariableOp_1:value:0Mresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
/residual_unit_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp8residual_unit_3_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_3/conv2d_10/Conv2DConv2D$residual_unit_2/Relu_1:activations:07residual_unit_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4residual_unit_3/batch_normalization_9/ReadVariableOpReadVariableOp=residual_unit_3_batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_9/ReadVariableOp_1ReadVariableOp?residual_unit_3_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Eresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpNresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPresidual_unit_3_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6residual_unit_3/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3)residual_unit_3/conv2d_10/Conv2D:output:0<residual_unit_3/batch_normalization_9/ReadVariableOp:value:0>residual_unit_3/batch_normalization_9/ReadVariableOp_1:value:0Mresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Oresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_3/addAddV2:residual_unit_3/batch_normalization_8/FusedBatchNormV3:y:0:residual_unit_3/batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������r
residual_unit_3/Relu_1Reluresidual_unit_3/add:z:0*
T0*0
_output_shapes
:�����������
/residual_unit_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp8residual_unit_4_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_4/conv2d_11/Conv2DConv2D$residual_unit_3/Relu_1:activations:07residual_unit_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_4/batch_normalization_10/ReadVariableOpReadVariableOp>residual_unit_4_batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_10/ReadVariableOp_1ReadVariableOp@residual_unit_4_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_4_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3)residual_unit_4/conv2d_11/Conv2D:output:0=residual_unit_4/batch_normalization_10/ReadVariableOp:value:0?residual_unit_4/batch_normalization_10/ReadVariableOp_1:value:0Nresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_4/ReluRelu;residual_unit_4/batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
/residual_unit_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp8residual_unit_4_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_4/conv2d_12/Conv2DConv2D"residual_unit_4/Relu:activations:07residual_unit_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_4/batch_normalization_11/ReadVariableOpReadVariableOp>residual_unit_4_batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_11/ReadVariableOp_1ReadVariableOp@residual_unit_4_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_4_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_4/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3)residual_unit_4/conv2d_12/Conv2D:output:0=residual_unit_4/batch_normalization_11/ReadVariableOp:value:0?residual_unit_4/batch_normalization_11/ReadVariableOp_1:value:0Nresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_4/addAddV2;residual_unit_4/batch_normalization_11/FusedBatchNormV3:y:0$residual_unit_3/Relu_1:activations:0*
T0*0
_output_shapes
:����������r
residual_unit_4/Relu_1Reluresidual_unit_4/add:z:0*
T0*0
_output_shapes
:�����������
/residual_unit_5/conv2d_13/Conv2D/ReadVariableOpReadVariableOp8residual_unit_5_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_5/conv2d_13/Conv2DConv2D$residual_unit_4/Relu_1:activations:07residual_unit_5/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_5/batch_normalization_12/ReadVariableOpReadVariableOp>residual_unit_5_batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_12/ReadVariableOp_1ReadVariableOp@residual_unit_5_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_5_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3)residual_unit_5/conv2d_13/Conv2D:output:0=residual_unit_5/batch_normalization_12/ReadVariableOp:value:0?residual_unit_5/batch_normalization_12/ReadVariableOp_1:value:0Nresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_5/ReluRelu;residual_unit_5/batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
/residual_unit_5/conv2d_14/Conv2D/ReadVariableOpReadVariableOp8residual_unit_5_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_5/conv2d_14/Conv2DConv2D"residual_unit_5/Relu:activations:07residual_unit_5/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_5/batch_normalization_13/ReadVariableOpReadVariableOp>residual_unit_5_batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_13/ReadVariableOp_1ReadVariableOp@residual_unit_5_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_5_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3)residual_unit_5/conv2d_14/Conv2D:output:0=residual_unit_5/batch_normalization_13/ReadVariableOp:value:0?residual_unit_5/batch_normalization_13/ReadVariableOp_1:value:0Nresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
/residual_unit_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp8residual_unit_5_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 residual_unit_5/conv2d_15/Conv2DConv2D$residual_unit_4/Relu_1:activations:07residual_unit_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5residual_unit_5/batch_normalization_14/ReadVariableOpReadVariableOp>residual_unit_5_batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_14/ReadVariableOp_1ReadVariableOp@residual_unit_5_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpOresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQresidual_unit_5_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7residual_unit_5/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3)residual_unit_5/conv2d_15/Conv2D:output:0=residual_unit_5/batch_normalization_14/ReadVariableOp:value:0?residual_unit_5/batch_normalization_14/ReadVariableOp_1:value:0Nresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Presidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
residual_unit_5/addAddV2;residual_unit_5/batch_normalization_13/FusedBatchNormV3:y:0;residual_unit_5/batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������r
residual_unit_5/Relu_1Reluresidual_unit_5/add:z:0*
T0*0
_output_shapes
:�����������
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d/MeanMean$residual_unit_5/Relu_1:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape&global_average_pooling2d/Mean:output:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������V
sampling/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0�
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*(
_output_shapes
:�����������
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*(
_output_shapes
:����������W
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
sampling/truedivRealDivdense_1/BiasAdd:output:0sampling/truediv/y:output:0*
T0*(
_output_shapes
:����������\
sampling/ExpExpsampling/truediv:z:0*
T0*(
_output_shapes
:����������t
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*(
_output_shapes
:����������r
sampling/addAddV2sampling/mul:z:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������j

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������b

Identity_2Identitysampling/add:z:0^NoOp*
T0*(
_output_shapes
:�����������%
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOpB^residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpD^residual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_11^residual_unit/batch_normalization/ReadVariableOp3^residual_unit/batch_normalization/ReadVariableOp_1D^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpF^residual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_13^residual_unit/batch_normalization_1/ReadVariableOp5^residual_unit/batch_normalization_1/ReadVariableOp_1-^residual_unit/conv2d_1/Conv2D/ReadVariableOp-^residual_unit/conv2d_2/Conv2D/ReadVariableOpF^residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpH^residual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_15^residual_unit_1/batch_normalization_2/ReadVariableOp7^residual_unit_1/batch_normalization_2/ReadVariableOp_1F^residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpH^residual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_15^residual_unit_1/batch_normalization_3/ReadVariableOp7^residual_unit_1/batch_normalization_3/ReadVariableOp_1F^residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpH^residual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_15^residual_unit_1/batch_normalization_4/ReadVariableOp7^residual_unit_1/batch_normalization_4/ReadVariableOp_1/^residual_unit_1/conv2d_3/Conv2D/ReadVariableOp/^residual_unit_1/conv2d_4/Conv2D/ReadVariableOp/^residual_unit_1/conv2d_5/Conv2D/ReadVariableOpF^residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpH^residual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_15^residual_unit_2/batch_normalization_5/ReadVariableOp7^residual_unit_2/batch_normalization_5/ReadVariableOp_1F^residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^residual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_15^residual_unit_2/batch_normalization_6/ReadVariableOp7^residual_unit_2/batch_normalization_6/ReadVariableOp_1/^residual_unit_2/conv2d_6/Conv2D/ReadVariableOp/^residual_unit_2/conv2d_7/Conv2D/ReadVariableOpF^residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpH^residual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_15^residual_unit_3/batch_normalization_7/ReadVariableOp7^residual_unit_3/batch_normalization_7/ReadVariableOp_1F^residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpH^residual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_15^residual_unit_3/batch_normalization_8/ReadVariableOp7^residual_unit_3/batch_normalization_8/ReadVariableOp_1F^residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpH^residual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_15^residual_unit_3/batch_normalization_9/ReadVariableOp7^residual_unit_3/batch_normalization_9/ReadVariableOp_10^residual_unit_3/conv2d_10/Conv2D/ReadVariableOp/^residual_unit_3/conv2d_8/Conv2D/ReadVariableOp/^residual_unit_3/conv2d_9/Conv2D/ReadVariableOpG^residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpI^residual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_16^residual_unit_4/batch_normalization_10/ReadVariableOp8^residual_unit_4/batch_normalization_10/ReadVariableOp_1G^residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpI^residual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_16^residual_unit_4/batch_normalization_11/ReadVariableOp8^residual_unit_4/batch_normalization_11/ReadVariableOp_10^residual_unit_4/conv2d_11/Conv2D/ReadVariableOp0^residual_unit_4/conv2d_12/Conv2D/ReadVariableOpG^residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpI^residual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_16^residual_unit_5/batch_normalization_12/ReadVariableOp8^residual_unit_5/batch_normalization_12/ReadVariableOp_1G^residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpI^residual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_16^residual_unit_5/batch_normalization_13/ReadVariableOp8^residual_unit_5/batch_normalization_13/ReadVariableOp_1G^residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpI^residual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_16^residual_unit_5/batch_normalization_14/ReadVariableOp8^residual_unit_5/batch_normalization_14/ReadVariableOp_10^residual_unit_5/conv2d_13/Conv2D/ReadVariableOp0^residual_unit_5/conv2d_14/Conv2D/ReadVariableOp0^residual_unit_5/conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2�
Aresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOpAresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Cresidual_unit/batch_normalization/FusedBatchNormV3/ReadVariableOp_12d
0residual_unit/batch_normalization/ReadVariableOp0residual_unit/batch_normalization/ReadVariableOp2h
2residual_unit/batch_normalization/ReadVariableOp_12residual_unit/batch_normalization/ReadVariableOp_12�
Cresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOpCresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Eresidual_unit/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12h
2residual_unit/batch_normalization_1/ReadVariableOp2residual_unit/batch_normalization_1/ReadVariableOp2l
4residual_unit/batch_normalization_1/ReadVariableOp_14residual_unit/batch_normalization_1/ReadVariableOp_12\
,residual_unit/conv2d_1/Conv2D/ReadVariableOp,residual_unit/conv2d_1/Conv2D/ReadVariableOp2\
,residual_unit/conv2d_2/Conv2D/ReadVariableOp,residual_unit/conv2d_2/Conv2D/ReadVariableOp2�
Eresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpEresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_2/ReadVariableOp4residual_unit_1/batch_normalization_2/ReadVariableOp2p
6residual_unit_1/batch_normalization_2/ReadVariableOp_16residual_unit_1/batch_normalization_2/ReadVariableOp_12�
Eresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpEresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_3/ReadVariableOp4residual_unit_1/batch_normalization_3/ReadVariableOp2p
6residual_unit_1/batch_normalization_3/ReadVariableOp_16residual_unit_1/batch_normalization_3/ReadVariableOp_12�
Eresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpEresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_1/batch_normalization_4/ReadVariableOp4residual_unit_1/batch_normalization_4/ReadVariableOp2p
6residual_unit_1/batch_normalization_4/ReadVariableOp_16residual_unit_1/batch_normalization_4/ReadVariableOp_12`
.residual_unit_1/conv2d_3/Conv2D/ReadVariableOp.residual_unit_1/conv2d_3/Conv2D/ReadVariableOp2`
.residual_unit_1/conv2d_4/Conv2D/ReadVariableOp.residual_unit_1/conv2d_4/Conv2D/ReadVariableOp2`
.residual_unit_1/conv2d_5/Conv2D/ReadVariableOp.residual_unit_1/conv2d_5/Conv2D/ReadVariableOp2�
Eresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpEresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_2/batch_normalization_5/ReadVariableOp4residual_unit_2/batch_normalization_5/ReadVariableOp2p
6residual_unit_2/batch_normalization_5/ReadVariableOp_16residual_unit_2/batch_normalization_5/ReadVariableOp_12�
Eresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_2/batch_normalization_6/ReadVariableOp4residual_unit_2/batch_normalization_6/ReadVariableOp2p
6residual_unit_2/batch_normalization_6/ReadVariableOp_16residual_unit_2/batch_normalization_6/ReadVariableOp_12`
.residual_unit_2/conv2d_6/Conv2D/ReadVariableOp.residual_unit_2/conv2d_6/Conv2D/ReadVariableOp2`
.residual_unit_2/conv2d_7/Conv2D/ReadVariableOp.residual_unit_2/conv2d_7/Conv2D/ReadVariableOp2�
Eresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOpEresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_3/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_3/batch_normalization_7/ReadVariableOp4residual_unit_3/batch_normalization_7/ReadVariableOp2p
6residual_unit_3/batch_normalization_7/ReadVariableOp_16residual_unit_3/batch_normalization_7/ReadVariableOp_12�
Eresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOpEresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_3/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_3/batch_normalization_8/ReadVariableOp4residual_unit_3/batch_normalization_8/ReadVariableOp2p
6residual_unit_3/batch_normalization_8/ReadVariableOp_16residual_unit_3/batch_normalization_8/ReadVariableOp_12�
Eresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOpEresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2�
Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Gresidual_unit_3/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12l
4residual_unit_3/batch_normalization_9/ReadVariableOp4residual_unit_3/batch_normalization_9/ReadVariableOp2p
6residual_unit_3/batch_normalization_9/ReadVariableOp_16residual_unit_3/batch_normalization_9/ReadVariableOp_12b
/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp/residual_unit_3/conv2d_10/Conv2D/ReadVariableOp2`
.residual_unit_3/conv2d_8/Conv2D/ReadVariableOp.residual_unit_3/conv2d_8/Conv2D/ReadVariableOp2`
.residual_unit_3/conv2d_9/Conv2D/ReadVariableOp.residual_unit_3/conv2d_9/Conv2D/ReadVariableOp2�
Fresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOpFresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_4/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_4/batch_normalization_10/ReadVariableOp5residual_unit_4/batch_normalization_10/ReadVariableOp2r
7residual_unit_4/batch_normalization_10/ReadVariableOp_17residual_unit_4/batch_normalization_10/ReadVariableOp_12�
Fresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOpFresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_4/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_4/batch_normalization_11/ReadVariableOp5residual_unit_4/batch_normalization_11/ReadVariableOp2r
7residual_unit_4/batch_normalization_11/ReadVariableOp_17residual_unit_4/batch_normalization_11/ReadVariableOp_12b
/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp/residual_unit_4/conv2d_11/Conv2D/ReadVariableOp2b
/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp/residual_unit_4/conv2d_12/Conv2D/ReadVariableOp2�
Fresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOpFresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_5/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_5/batch_normalization_12/ReadVariableOp5residual_unit_5/batch_normalization_12/ReadVariableOp2r
7residual_unit_5/batch_normalization_12/ReadVariableOp_17residual_unit_5/batch_normalization_12/ReadVariableOp_12�
Fresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOpFresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_5/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_5/batch_normalization_13/ReadVariableOp5residual_unit_5/batch_normalization_13/ReadVariableOp2r
7residual_unit_5/batch_normalization_13/ReadVariableOp_17residual_unit_5/batch_normalization_13/ReadVariableOp_12�
Fresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOpFresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2�
Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Hresidual_unit_5/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12n
5residual_unit_5/batch_normalization_14/ReadVariableOp5residual_unit_5/batch_normalization_14/ReadVariableOp2r
7residual_unit_5/batch_normalization_14/ReadVariableOp_17residual_unit_5/batch_normalization_14/ReadVariableOp_12b
/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp/residual_unit_5/conv2d_13/Conv2D/ReadVariableOp2b
/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp/residual_unit_5/conv2d_14/Conv2D/ReadVariableOp2b
/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp/residual_unit_5/conv2d_15/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128017066

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�[
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128019327

inputsB
'conv2d_3_conv2d_readvariableop_resource:@�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��<
-batch_normalization_3_readvariableop_resource:	�>
/batch_normalization_3_readvariableop_1_resource:	�M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�B
'conv2d_5_conv2d_readvariableop_resource:@�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2DRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������  @: : : : : : : : : : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023626

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�j
�
D__inference_model_layer_call_and_return_conditional_losses_128018589

inputs*
conv2d_128018057:@1
residual_unit_128018110:@@%
residual_unit_128018112:@%
residual_unit_128018114:@%
residual_unit_128018116:@%
residual_unit_128018118:@1
residual_unit_128018120:@@%
residual_unit_128018122:@%
residual_unit_128018124:@%
residual_unit_128018126:@%
residual_unit_128018128:@4
residual_unit_1_128018190:@�(
residual_unit_1_128018192:	�(
residual_unit_1_128018194:	�(
residual_unit_1_128018196:	�(
residual_unit_1_128018198:	�5
residual_unit_1_128018200:��(
residual_unit_1_128018202:	�(
residual_unit_1_128018204:	�(
residual_unit_1_128018206:	�(
residual_unit_1_128018208:	�4
residual_unit_1_128018210:@�(
residual_unit_1_128018212:	�(
residual_unit_1_128018214:	�(
residual_unit_1_128018216:	�(
residual_unit_1_128018218:	�5
residual_unit_2_128018263:��(
residual_unit_2_128018265:	�(
residual_unit_2_128018267:	�(
residual_unit_2_128018269:	�(
residual_unit_2_128018271:	�5
residual_unit_2_128018273:��(
residual_unit_2_128018275:	�(
residual_unit_2_128018277:	�(
residual_unit_2_128018279:	�(
residual_unit_2_128018281:	�5
residual_unit_3_128018343:��(
residual_unit_3_128018345:	�(
residual_unit_3_128018347:	�(
residual_unit_3_128018349:	�(
residual_unit_3_128018351:	�5
residual_unit_3_128018353:��(
residual_unit_3_128018355:	�(
residual_unit_3_128018357:	�(
residual_unit_3_128018359:	�(
residual_unit_3_128018361:	�5
residual_unit_3_128018363:��(
residual_unit_3_128018365:	�(
residual_unit_3_128018367:	�(
residual_unit_3_128018369:	�(
residual_unit_3_128018371:	�5
residual_unit_4_128018416:��(
residual_unit_4_128018418:	�(
residual_unit_4_128018420:	�(
residual_unit_4_128018422:	�(
residual_unit_4_128018424:	�5
residual_unit_4_128018426:��(
residual_unit_4_128018428:	�(
residual_unit_4_128018430:	�(
residual_unit_4_128018432:	�(
residual_unit_4_128018434:	�5
residual_unit_5_128018496:��(
residual_unit_5_128018498:	�(
residual_unit_5_128018500:	�(
residual_unit_5_128018502:	�(
residual_unit_5_128018504:	�5
residual_unit_5_128018506:��(
residual_unit_5_128018508:	�(
residual_unit_5_128018510:	�(
residual_unit_5_128018512:	�(
residual_unit_5_128018514:	�5
residual_unit_5_128018516:��(
residual_unit_5_128018518:	�(
residual_unit_5_128018520:	�(
residual_unit_5_128018522:	�(
residual_unit_5_128018524:	�#
dense_128018547:
��
dense_128018549:	�%
dense_1_128018563:
�� 
dense_1_128018565:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_128018057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_128018056�
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_128018065�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128017066�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_128018110residual_unit_128018112residual_unit_128018114residual_unit_128018116residual_unit_128018118residual_unit_128018120residual_unit_128018122residual_unit_128018124residual_unit_128018126residual_unit_128018128*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_residual_unit_layer_call_and_return_conditional_losses_128018109�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_128018190residual_unit_1_128018192residual_unit_1_128018194residual_unit_1_128018196residual_unit_1_128018198residual_unit_1_128018200residual_unit_1_128018202residual_unit_1_128018204residual_unit_1_128018206residual_unit_1_128018208residual_unit_1_128018210residual_unit_1_128018212residual_unit_1_128018214residual_unit_1_128018216residual_unit_1_128018218*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128018189�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_128018263residual_unit_2_128018265residual_unit_2_128018267residual_unit_2_128018269residual_unit_2_128018271residual_unit_2_128018273residual_unit_2_128018275residual_unit_2_128018277residual_unit_2_128018279residual_unit_2_128018281*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128018262�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_128018343residual_unit_3_128018345residual_unit_3_128018347residual_unit_3_128018349residual_unit_3_128018351residual_unit_3_128018353residual_unit_3_128018355residual_unit_3_128018357residual_unit_3_128018359residual_unit_3_128018361residual_unit_3_128018363residual_unit_3_128018365residual_unit_3_128018367residual_unit_3_128018369residual_unit_3_128018371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128018342�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_128018416residual_unit_4_128018418residual_unit_4_128018420residual_unit_4_128018422residual_unit_4_128018424residual_unit_4_128018426residual_unit_4_128018428residual_unit_4_128018430residual_unit_4_128018432residual_unit_4_128018434*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018415�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_128018496residual_unit_5_128018498residual_unit_5_128018500residual_unit_5_128018502residual_unit_5_128018504residual_unit_5_128018506residual_unit_5_128018508residual_unit_5_128018510residual_unit_5_128018512residual_unit_5_128018514residual_unit_5_128018516residual_unit_5_128018518residual_unit_5_128018520residual_unit_5_128018522residual_unit_5_128018524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018495�
(global_average_pooling2d/PartitionedCallPartitionedCall0residual_unit_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128018039�
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_128018534�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_128018547dense_128018549*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_128018546�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_128018563dense_1_128018565*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_128018562�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_128018584v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall(^residual_unit_1/StatefulPartitionedCall(^residual_unit_2/StatefulPartitionedCall(^residual_unit_3/StatefulPartitionedCall(^residual_unit_4/StatefulPartitionedCall(^residual_unit_5/StatefulPartitionedCall!^sampling/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall2R
'residual_unit_1/StatefulPartitionedCall'residual_unit_1/StatefulPartitionedCall2R
'residual_unit_2/StatefulPartitionedCall'residual_unit_2/StatefulPartitionedCall2R
'residual_unit_3/StatefulPartitionedCall'residual_unit_3/StatefulPartitionedCall2R
'residual_unit_4/StatefulPartitionedCall'residual_unit_4/StatefulPartitionedCall2R
'residual_unit_5/StatefulPartitionedCall'residual_unit_5/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
3__inference_residual_unit_2_layer_call_fn_128022046

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128019208x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128017603

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�?
�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022128

inputsC
'conv2d_6_conv2d_readvariableop_resource:��<
-batch_normalization_5_readvariableop_resource:	�>
/batch_normalization_5_readvariableop_1_resource:	�M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_7_conv2d_readvariableop_resource:��<
-batch_normalization_6_readvariableop_resource:	�>
/batch_normalization_6_readvariableop_1_resource:	�M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2DRelu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1^conv2d_6/Conv2D/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022776

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�>
�

L__inference_residual_unit_layer_call_and_return_conditional_losses_128019429

inputsA
'conv2d_1_conv2d_readvariableop_resource:@@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:���������  @Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:���������  @k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:���������  @�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  @: : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
3__inference_residual_unit_1_layer_call_fn_128021880

inputs"
unknown:@�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�$
	unknown_9:@�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128019327x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������  @: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
+__inference_dense_1_layer_call_fn_128022682

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_128018562p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_activation_layer_call_fn_128021663

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_128018065h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�&
�
)__inference_model_layer_call_fn_128020153
input_1!
unknown:@#
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@%

unknown_10:@�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:@�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�&

unknown_25:��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�&

unknown_30:��

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�&

unknown_35:��

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�&

unknown_40:��

unknown_41:	�

unknown_42:	�

unknown_43:	�

unknown_44:	�&

unknown_45:��

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�&

unknown_50:��

unknown_51:	�

unknown_52:	�

unknown_53:	�

unknown_54:	�&

unknown_55:��

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�&

unknown_60:��

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�&

unknown_65:��

unknown_66:	�

unknown_67:	�

unknown_68:	�

unknown_69:	�&

unknown_70:��

unknown_71:	�

unknown_72:	�

unknown_73:	�

unknown_74:	�

unknown_75:
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*T
_read_only_resource_inputs6
42	 !"%&'*+,/014569:;>?@CDEHIJMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_128019817p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�	
�
9__inference_batch_normalization_8_layer_call_fn_128023223

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128017603�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128017378

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022944

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�G
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018495

inputsD
(conv2d_13_conv2d_readvariableop_resource:��=
.batch_normalization_12_readvariableop_resource:	�?
0batch_normalization_12_readvariableop_1_resource:	�N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_14_conv2d_readvariableop_resource:��=
.batch_normalization_13_readvariableop_resource:	�?
0batch_normalization_13_readvariableop_1_resource:	�N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_15_conv2d_readvariableop_resource:��=
.batch_normalization_14_readvariableop_resource:	�?
0batch_normalization_14_readvariableop_1_resource:	�N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_12/FusedBatchNormV3/ReadVariableOp�8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_12/ReadVariableOp�'batch_normalization_12/ReadVariableOp_1�6batch_normalization_13/FusedBatchNormV3/ReadVariableOp�8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_13/ReadVariableOp�'batch_normalization_13/ReadVariableOp_1�6batch_normalization_14/FusedBatchNormV3/ReadVariableOp�8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_14/ReadVariableOp�'batch_normalization_14/ReadVariableOp_1�conv2d_13/Conv2D/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_13/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( t
ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_14/Conv2DConv2DRelu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_14/Conv2D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_15/Conv2D:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
addAddV2+batch_normalization_13/FusedBatchNormV3:y:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1 ^conv2d_13/Conv2D/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023502

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_128018534

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128017347

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�[
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022314

inputsC
'conv2d_8_conv2d_readvariableop_resource:��<
-batch_normalization_7_readvariableop_resource:	�>
/batch_normalization_7_readvariableop_1_resource:	�M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��<
-batch_normalization_8_readvariableop_resource:	�>
/batch_normalization_8_readvariableop_1_resource:	�M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_7/AssignNewValue�&batch_normalization_7/AssignNewValue_1�5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�$batch_normalization_8/AssignNewValue�&batch_normalization_8/AssignNewValue_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�$batch_normalization_9/AssignNewValue�&batch_normalization_9/AssignNewValue_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�conv2d_10/Conv2D/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2DConv2DRelu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_9/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_10/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
addAddV2*batch_normalization_8/FusedBatchNormV3:y:0*batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_10/Conv2D/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128017314

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023520

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�&
�
)__inference_model_layer_call_fn_128018756
input_1!
unknown:@#
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@%

unknown_10:@�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:@�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�&

unknown_25:��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�&

unknown_30:��

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�&

unknown_35:��

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�&

unknown_40:��

unknown_41:	�

unknown_42:	�

unknown_43:	�

unknown_44:	�&

unknown_45:��

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�&

unknown_50:��

unknown_51:	�

unknown_52:	�

unknown_53:	�

unknown_54:	�&

unknown_55:��

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�&

unknown_60:��

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�&

unknown_65:��

unknown_66:	�

unknown_67:	�

unknown_68:	�

unknown_69:	�&

unknown_70:��

unknown_71:	�

unknown_72:	�

unknown_73:	�

unknown_74:	�

unknown_75:
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_128018589p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128017698

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128017186

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�?
�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128019208

inputsC
'conv2d_6_conv2d_readvariableop_resource:��<
-batch_normalization_5_readvariableop_resource:	�>
/batch_normalization_5_readvariableop_1_resource:	�M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_7_conv2d_readvariableop_resource:��<
-batch_normalization_6_readvariableop_resource:	�>
/batch_normalization_6_readvariableop_1_resource:	�M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2DRelu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape({
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1^conv2d_6/Conv2D/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023130

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128017634

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�]
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018885

inputsD
(conv2d_13_conv2d_readvariableop_resource:��=
.batch_normalization_12_readvariableop_resource:	�?
0batch_normalization_12_readvariableop_1_resource:	�N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_14_conv2d_readvariableop_resource:��=
.batch_normalization_13_readvariableop_resource:	�?
0batch_normalization_13_readvariableop_1_resource:	�N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_15_conv2d_readvariableop_resource:��=
.batch_normalization_14_readvariableop_resource:	�?
0batch_normalization_14_readvariableop_1_resource:	�N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_12/AssignNewValue�'batch_normalization_12/AssignNewValue_1�6batch_normalization_12/FusedBatchNormV3/ReadVariableOp�8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_12/ReadVariableOp�'batch_normalization_12/ReadVariableOp_1�%batch_normalization_13/AssignNewValue�'batch_normalization_13/AssignNewValue_1�6batch_normalization_13/FusedBatchNormV3/ReadVariableOp�8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_13/ReadVariableOp�'batch_normalization_13/ReadVariableOp_1�%batch_normalization_14/AssignNewValue�'batch_normalization_14/AssignNewValue_1�6batch_normalization_14/FusedBatchNormV3/ReadVariableOp�8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_14/ReadVariableOp�'batch_normalization_14/ReadVariableOp_1�conv2d_13/Conv2D/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_13/Conv2DConv2Dinputs'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_13/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
ReluRelu+batch_normalization_12/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_14/Conv2DConv2DRelu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_14/Conv2D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_15/Conv2D:output:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
addAddV2+batch_normalization_13/FusedBatchNormV3:y:0+batch_normalization_14/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1 ^conv2d_13/Conv2D/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�j
�
D__inference_model_layer_call_and_return_conditional_losses_128019817

inputs*
conv2d_128019640:@1
residual_unit_128019645:@@%
residual_unit_128019647:@%
residual_unit_128019649:@%
residual_unit_128019651:@%
residual_unit_128019653:@1
residual_unit_128019655:@@%
residual_unit_128019657:@%
residual_unit_128019659:@%
residual_unit_128019661:@%
residual_unit_128019663:@4
residual_unit_1_128019666:@�(
residual_unit_1_128019668:	�(
residual_unit_1_128019670:	�(
residual_unit_1_128019672:	�(
residual_unit_1_128019674:	�5
residual_unit_1_128019676:��(
residual_unit_1_128019678:	�(
residual_unit_1_128019680:	�(
residual_unit_1_128019682:	�(
residual_unit_1_128019684:	�4
residual_unit_1_128019686:@�(
residual_unit_1_128019688:	�(
residual_unit_1_128019690:	�(
residual_unit_1_128019692:	�(
residual_unit_1_128019694:	�5
residual_unit_2_128019697:��(
residual_unit_2_128019699:	�(
residual_unit_2_128019701:	�(
residual_unit_2_128019703:	�(
residual_unit_2_128019705:	�5
residual_unit_2_128019707:��(
residual_unit_2_128019709:	�(
residual_unit_2_128019711:	�(
residual_unit_2_128019713:	�(
residual_unit_2_128019715:	�5
residual_unit_3_128019718:��(
residual_unit_3_128019720:	�(
residual_unit_3_128019722:	�(
residual_unit_3_128019724:	�(
residual_unit_3_128019726:	�5
residual_unit_3_128019728:��(
residual_unit_3_128019730:	�(
residual_unit_3_128019732:	�(
residual_unit_3_128019734:	�(
residual_unit_3_128019736:	�5
residual_unit_3_128019738:��(
residual_unit_3_128019740:	�(
residual_unit_3_128019742:	�(
residual_unit_3_128019744:	�(
residual_unit_3_128019746:	�5
residual_unit_4_128019749:��(
residual_unit_4_128019751:	�(
residual_unit_4_128019753:	�(
residual_unit_4_128019755:	�(
residual_unit_4_128019757:	�5
residual_unit_4_128019759:��(
residual_unit_4_128019761:	�(
residual_unit_4_128019763:	�(
residual_unit_4_128019765:	�(
residual_unit_4_128019767:	�5
residual_unit_5_128019770:��(
residual_unit_5_128019772:	�(
residual_unit_5_128019774:	�(
residual_unit_5_128019776:	�(
residual_unit_5_128019778:	�5
residual_unit_5_128019780:��(
residual_unit_5_128019782:	�(
residual_unit_5_128019784:	�(
residual_unit_5_128019786:	�(
residual_unit_5_128019788:	�5
residual_unit_5_128019790:��(
residual_unit_5_128019792:	�(
residual_unit_5_128019794:	�(
residual_unit_5_128019796:	�(
residual_unit_5_128019798:	�#
dense_128019803:
��
dense_128019805:	�%
dense_1_128019808:
�� 
dense_1_128019810:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_128019640*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_128018056�
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_128018065�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128017066�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_128019645residual_unit_128019647residual_unit_128019649residual_unit_128019651residual_unit_128019653residual_unit_128019655residual_unit_128019657residual_unit_128019659residual_unit_128019661residual_unit_128019663*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_residual_unit_layer_call_and_return_conditional_losses_128019429�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_128019666residual_unit_1_128019668residual_unit_1_128019670residual_unit_1_128019672residual_unit_1_128019674residual_unit_1_128019676residual_unit_1_128019678residual_unit_1_128019680residual_unit_1_128019682residual_unit_1_128019684residual_unit_1_128019686residual_unit_1_128019688residual_unit_1_128019690residual_unit_1_128019692residual_unit_1_128019694*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128019327�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_128019697residual_unit_2_128019699residual_unit_2_128019701residual_unit_2_128019703residual_unit_2_128019705residual_unit_2_128019707residual_unit_2_128019709residual_unit_2_128019711residual_unit_2_128019713residual_unit_2_128019715*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128019208�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_128019718residual_unit_3_128019720residual_unit_3_128019722residual_unit_3_128019724residual_unit_3_128019726residual_unit_3_128019728residual_unit_3_128019730residual_unit_3_128019732residual_unit_3_128019734residual_unit_3_128019736residual_unit_3_128019738residual_unit_3_128019740residual_unit_3_128019742residual_unit_3_128019744residual_unit_3_128019746*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128019106�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_128019749residual_unit_4_128019751residual_unit_4_128019753residual_unit_4_128019755residual_unit_4_128019757residual_unit_4_128019759residual_unit_4_128019761residual_unit_4_128019763residual_unit_4_128019765residual_unit_4_128019767*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018987�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_128019770residual_unit_5_128019772residual_unit_5_128019774residual_unit_5_128019776residual_unit_5_128019778residual_unit_5_128019780residual_unit_5_128019782residual_unit_5_128019784residual_unit_5_128019786residual_unit_5_128019788residual_unit_5_128019790residual_unit_5_128019792residual_unit_5_128019794residual_unit_5_128019796residual_unit_5_128019798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018885�
(global_average_pooling2d/PartitionedCallPartitionedCall0residual_unit_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128018039�
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_128018534�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_128019803dense_128019805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_128018546�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_128019808dense_1_128019810*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_128018562�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_128018584v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall(^residual_unit_1/StatefulPartitionedCall(^residual_unit_2/StatefulPartitionedCall(^residual_unit_3/StatefulPartitionedCall(^residual_unit_4/StatefulPartitionedCall(^residual_unit_5/StatefulPartitionedCall!^sampling/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall2R
'residual_unit_1/StatefulPartitionedCall'residual_unit_1/StatefulPartitionedCall2R
'residual_unit_2/StatefulPartitionedCall'residual_unit_2/StatefulPartitionedCall2R
'residual_unit_3/StatefulPartitionedCall'residual_unit_3/StatefulPartitionedCall2R
'residual_unit_4/StatefulPartitionedCall'residual_unit_4/StatefulPartitionedCall2R
'residual_unit_5/StatefulPartitionedCall'residual_unit_5/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_3_layer_call_fn_128022913

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128017283�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
D__inference_dense_layer_call_and_return_conditional_losses_128022673

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128017506

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128017283

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_4_layer_call_fn_128022975

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128017347�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023024

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023396

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_layer_call_and_return_conditional_losses_128021658

inputs8
conv2d_readvariableop_resource:@
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������@@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_12_layer_call_fn_128023471

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128017859�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128017667

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
3__inference_residual_unit_4_layer_call_fn_128022339

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018415x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�F
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128018342

inputsC
'conv2d_8_conv2d_readvariableop_resource:��<
-batch_normalization_7_readvariableop_resource:	�>
/batch_normalization_7_readvariableop_1_resource:	�M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��<
-batch_normalization_8_readvariableop_resource:	�>
/batch_normalization_8_readvariableop_1_resource:	�M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�conv2d_10/Conv2D/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( s
ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2DConv2DRelu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_9/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_10/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
addAddV2*batch_normalization_8/FusedBatchNormV3:y:0*batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_10/Conv2D/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_12_layer_call_fn_128023484

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128017890�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
3__inference_residual_unit_5_layer_call_fn_128022481

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018495x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022882

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_2_layer_call_fn_128022851

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128017219�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_7_layer_call_fn_128023174

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128017570�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_1_layer_call_fn_128022802

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128017186�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_128022654

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�/
�	
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021769

inputsA
'conv2d_1_conv2d_readvariableop_resource:@@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( p
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( z
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:���������  @Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:���������  @k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:���������  @�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  @: : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�>
�

L__inference_residual_unit_layer_call_and_return_conditional_losses_128021810

inputsA
'conv2d_1_conv2d_readvariableop_resource:@@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������  @�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_2/Conv2DConv2DRelu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(z
addAddV2*batch_normalization_1/FusedBatchNormV3:y:0inputs*
T0*/
_output_shapes
:���������  @Q
Relu_1Reluadd:z:0*
T0*/
_output_shapes
:���������  @k
IdentityIdentityRelu_1:activations:0^NoOp*
T0*/
_output_shapes
:���������  @�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  @: : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128017475

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�[
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021996

inputsB
'conv2d_3_conv2d_readvariableop_resource:@�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��<
-batch_normalization_3_readvariableop_resource:	�>
/batch_normalization_3_readvariableop_1_resource:	�M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�B
'conv2d_5_conv2d_readvariableop_resource:@�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(s
ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2DRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������  @: : : : : : : : : : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�1
�

N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022405

inputsD
(conv2d_11_conv2d_readvariableop_resource:��=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_12_conv2d_readvariableop_resource:��=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�conv2d_11/Conv2D/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_11/Conv2D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( t
ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_12/Conv2DConv2DRelu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_12/Conv2D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( |
addAddV2+batch_normalization_11/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1 ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
)__inference_model_layer_call_fn_128020853

inputs!
unknown:@#
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@%

unknown_10:@�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:@�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�&

unknown_25:��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�&

unknown_30:��

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�&

unknown_35:��

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�&

unknown_40:��

unknown_41:	�

unknown_42:	�

unknown_43:	�

unknown_44:	�&

unknown_45:��

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�&

unknown_50:��

unknown_51:	�

unknown_52:	�

unknown_53:	�

unknown_54:	�&

unknown_55:��

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�&

unknown_60:��

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�&

unknown_65:��

unknown_66:	�

unknown_67:	�

unknown_68:	�

unknown_69:	�&

unknown_70:��

unknown_71:	�

unknown_72:	�

unknown_73:	�

unknown_74:	�

unknown_75:
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_128018589p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_4_layer_call_fn_128022988

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128017378�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�j
�
D__inference_model_layer_call_and_return_conditional_losses_128020333
input_1*
conv2d_128020156:@1
residual_unit_128020161:@@%
residual_unit_128020163:@%
residual_unit_128020165:@%
residual_unit_128020167:@%
residual_unit_128020169:@1
residual_unit_128020171:@@%
residual_unit_128020173:@%
residual_unit_128020175:@%
residual_unit_128020177:@%
residual_unit_128020179:@4
residual_unit_1_128020182:@�(
residual_unit_1_128020184:	�(
residual_unit_1_128020186:	�(
residual_unit_1_128020188:	�(
residual_unit_1_128020190:	�5
residual_unit_1_128020192:��(
residual_unit_1_128020194:	�(
residual_unit_1_128020196:	�(
residual_unit_1_128020198:	�(
residual_unit_1_128020200:	�4
residual_unit_1_128020202:@�(
residual_unit_1_128020204:	�(
residual_unit_1_128020206:	�(
residual_unit_1_128020208:	�(
residual_unit_1_128020210:	�5
residual_unit_2_128020213:��(
residual_unit_2_128020215:	�(
residual_unit_2_128020217:	�(
residual_unit_2_128020219:	�(
residual_unit_2_128020221:	�5
residual_unit_2_128020223:��(
residual_unit_2_128020225:	�(
residual_unit_2_128020227:	�(
residual_unit_2_128020229:	�(
residual_unit_2_128020231:	�5
residual_unit_3_128020234:��(
residual_unit_3_128020236:	�(
residual_unit_3_128020238:	�(
residual_unit_3_128020240:	�(
residual_unit_3_128020242:	�5
residual_unit_3_128020244:��(
residual_unit_3_128020246:	�(
residual_unit_3_128020248:	�(
residual_unit_3_128020250:	�(
residual_unit_3_128020252:	�5
residual_unit_3_128020254:��(
residual_unit_3_128020256:	�(
residual_unit_3_128020258:	�(
residual_unit_3_128020260:	�(
residual_unit_3_128020262:	�5
residual_unit_4_128020265:��(
residual_unit_4_128020267:	�(
residual_unit_4_128020269:	�(
residual_unit_4_128020271:	�(
residual_unit_4_128020273:	�5
residual_unit_4_128020275:��(
residual_unit_4_128020277:	�(
residual_unit_4_128020279:	�(
residual_unit_4_128020281:	�(
residual_unit_4_128020283:	�5
residual_unit_5_128020286:��(
residual_unit_5_128020288:	�(
residual_unit_5_128020290:	�(
residual_unit_5_128020292:	�(
residual_unit_5_128020294:	�5
residual_unit_5_128020296:��(
residual_unit_5_128020298:	�(
residual_unit_5_128020300:	�(
residual_unit_5_128020302:	�(
residual_unit_5_128020304:	�5
residual_unit_5_128020306:��(
residual_unit_5_128020308:	�(
residual_unit_5_128020310:	�(
residual_unit_5_128020312:	�(
residual_unit_5_128020314:	�#
dense_128020319:
��
dense_128020321:	�%
dense_1_128020324:
�� 
dense_1_128020326:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_128020156*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_128018056�
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_128018065�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128017066�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_128020161residual_unit_128020163residual_unit_128020165residual_unit_128020167residual_unit_128020169residual_unit_128020171residual_unit_128020173residual_unit_128020175residual_unit_128020177residual_unit_128020179*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_residual_unit_layer_call_and_return_conditional_losses_128018109�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_128020182residual_unit_1_128020184residual_unit_1_128020186residual_unit_1_128020188residual_unit_1_128020190residual_unit_1_128020192residual_unit_1_128020194residual_unit_1_128020196residual_unit_1_128020198residual_unit_1_128020200residual_unit_1_128020202residual_unit_1_128020204residual_unit_1_128020206residual_unit_1_128020208residual_unit_1_128020210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128018189�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_128020213residual_unit_2_128020215residual_unit_2_128020217residual_unit_2_128020219residual_unit_2_128020221residual_unit_2_128020223residual_unit_2_128020225residual_unit_2_128020227residual_unit_2_128020229residual_unit_2_128020231*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128018262�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_128020234residual_unit_3_128020236residual_unit_3_128020238residual_unit_3_128020240residual_unit_3_128020242residual_unit_3_128020244residual_unit_3_128020246residual_unit_3_128020248residual_unit_3_128020250residual_unit_3_128020252residual_unit_3_128020254residual_unit_3_128020256residual_unit_3_128020258residual_unit_3_128020260residual_unit_3_128020262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128018342�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_128020265residual_unit_4_128020267residual_unit_4_128020269residual_unit_4_128020271residual_unit_4_128020273residual_unit_4_128020275residual_unit_4_128020277residual_unit_4_128020279residual_unit_4_128020281residual_unit_4_128020283*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018415�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_128020286residual_unit_5_128020288residual_unit_5_128020290residual_unit_5_128020292residual_unit_5_128020294residual_unit_5_128020296residual_unit_5_128020298residual_unit_5_128020300residual_unit_5_128020302residual_unit_5_128020304residual_unit_5_128020306residual_unit_5_128020308residual_unit_5_128020310residual_unit_5_128020312residual_unit_5_128020314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018495�
(global_average_pooling2d/PartitionedCallPartitionedCall0residual_unit_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128018039�
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_128018534�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_128020319dense_128020321*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_128018546�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_128020324dense_1_128020326*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_128018562�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_128018584v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall(^residual_unit_1/StatefulPartitionedCall(^residual_unit_2/StatefulPartitionedCall(^residual_unit_3/StatefulPartitionedCall(^residual_unit_4/StatefulPartitionedCall(^residual_unit_5/StatefulPartitionedCall!^sampling/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall2R
'residual_unit_1/StatefulPartitionedCall'residual_unit_1/StatefulPartitionedCall2R
'residual_unit_2/StatefulPartitionedCall'residual_unit_2/StatefulPartitionedCall2R
'residual_unit_3/StatefulPartitionedCall'residual_unit_3/StatefulPartitionedCall2R
'residual_unit_4/StatefulPartitionedCall'residual_unit_4/StatefulPartitionedCall2R
'residual_unit_5/StatefulPartitionedCall'residual_unit_5/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
X
<__inference_global_average_pooling2d_layer_call_fn_128022637

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128018039i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128017923

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_layer_call_fn_128022727

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128017091�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�&
�
'__inference_signature_wrapper_128020684
input_1!
unknown:@#
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@%

unknown_10:@�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:@�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�&

unknown_25:��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�&

unknown_30:��

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�&

unknown_35:��

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�&

unknown_40:��

unknown_41:	�

unknown_42:	�

unknown_43:	�

unknown_44:	�&

unknown_45:��

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�&

unknown_50:��

unknown_51:	�

unknown_52:	�

unknown_53:	�

unknown_54:	�&

unknown_55:��

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�&

unknown_60:��

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�&

unknown_65:��

unknown_66:	�

unknown_67:	�

unknown_68:	�

unknown_69:	�&

unknown_70:��

unknown_71:	�

unknown_72:	�

unknown_73:	�

unknown_74:	�

unknown_75:
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__wrapped_model_128017057p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128018018

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128017890

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
)__inference_dense_layer_call_fn_128022663

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_128018546p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�@
�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018987

inputsD
(conv2d_11_conv2d_readvariableop_resource:��=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_12_conv2d_readvariableop_resource:��=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_10/AssignNewValue�'batch_normalization_10/AssignNewValue_1�6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�%batch_normalization_11/AssignNewValue�'batch_normalization_11/AssignNewValue_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�conv2d_11/Conv2D/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_11/Conv2D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_12/Conv2DConv2DRelu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_12/Conv2D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(|
addAddV2+batch_normalization_11/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1 ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_13_layer_call_fn_128023533

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128017923�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_2_layer_call_fn_128022864

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128017250�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
3__inference_residual_unit_5_layer_call_fn_128022516

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018885x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_1_layer_call_and_return_conditional_losses_128022692

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023316

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_13_layer_call_fn_128023546

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128017954�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_8_layer_call_fn_128023236

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128017634�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
3__inference_residual_unit_1_layer_call_fn_128021845

inputs"
unknown:@�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�$
	unknown_9:@�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128018189x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������  @: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_9_layer_call_fn_128023285

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128017667�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023582

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128017122

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
F__inference_dense_1_layer_call_and_return_conditional_losses_128018562

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128017987

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023006

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
D__inference_dense_layer_call_and_return_conditional_losses_128018546

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
1__inference_residual_unit_layer_call_fn_128021703

inputs!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@#
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_residual_unit_layer_call_and_return_conditional_losses_128018109w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  @: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128017539

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023458

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�F
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128018189

inputsB
'conv2d_3_conv2d_readvariableop_resource:@�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_4_conv2d_readvariableop_resource:��<
-batch_normalization_3_readvariableop_resource:	�>
/batch_normalization_3_readvariableop_1_resource:	�M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	�B
'conv2d_5_conv2d_readvariableop_resource:@�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( s
ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2DRelu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
addAddV2*batch_normalization_3/FusedBatchNormV3:y:0*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������  @: : : : : : : : : : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  @
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_5_layer_call_fn_128023037

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128017411�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�0
�	
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022087

inputsC
'conv2d_6_conv2d_readvariableop_resource:��<
-batch_normalization_5_readvariableop_resource:	�>
/batch_normalization_5_readvariableop_1_resource:	�M
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_7_conv2d_readvariableop_resource:��<
-batch_normalization_6_readvariableop_resource:	�>
/batch_normalization_6_readvariableop_1_resource:	�M
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( s
ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_7/Conv2DConv2DRelu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( {
addAddV2*batch_normalization_6/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1^conv2d_6/Conv2D/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128021678

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_10_layer_call_fn_128023360

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128017762�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023378

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128017411

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023086

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�F
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022256

inputsC
'conv2d_8_conv2d_readvariableop_resource:��<
-batch_normalization_7_readvariableop_resource:	�>
/batch_normalization_7_readvariableop_1_resource:	�M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��<
-batch_normalization_8_readvariableop_resource:	�>
/batch_normalization_8_readvariableop_1_resource:	�M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��<
-batch_normalization_9_readvariableop_resource:	�>
/batch_normalization_9_readvariableop_1_resource:	�M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�5batch_normalization_8/FusedBatchNormV3/ReadVariableOp�7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_8/ReadVariableOp�&batch_normalization_8/ReadVariableOp_1�5batch_normalization_9/FusedBatchNormV3/ReadVariableOp�7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_9/ReadVariableOp�&batch_normalization_9/ReadVariableOp_1�conv2d_10/Conv2D/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( s
ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_9/Conv2DConv2DRelu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_9/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_10/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
addAddV2*batch_normalization_8/FusedBatchNormV3:y:0*batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_10/Conv2D/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128017859

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023440

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023254

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
3__inference_residual_unit_3_layer_call_fn_128022198

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128019106x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_1_layer_call_fn_128022789

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128017155�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_9_layer_call_fn_128023298

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128017698�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_6_layer_call_fn_128023112

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128017506�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�@
�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022446

inputsD
(conv2d_11_conv2d_readvariableop_resource:��=
.batch_normalization_10_readvariableop_resource:	�?
0batch_normalization_10_readvariableop_1_resource:	�N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_12_conv2d_readvariableop_resource:��=
.batch_normalization_11_readvariableop_resource:	�?
0batch_normalization_11_readvariableop_1_resource:	�N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_10/AssignNewValue�'batch_normalization_10/AssignNewValue_1�6batch_normalization_10/FusedBatchNormV3/ReadVariableOp�8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_10/ReadVariableOp�'batch_normalization_10/ReadVariableOp_1�%batch_normalization_11/AssignNewValue�'batch_normalization_11/AssignNewValue_1�6batch_normalization_11/FusedBatchNormV3/ReadVariableOp�8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_11/ReadVariableOp�'batch_normalization_11/ReadVariableOp_1�conv2d_11/Conv2D/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_11/Conv2D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(t
ReluRelu+batch_normalization_10/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_12/Conv2DConv2DRelu:activations:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_12/Conv2D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(|
addAddV2+batch_normalization_11/FusedBatchNormV3:y:0inputs*
T0*0
_output_shapes
:����������R
Relu_1Reluadd:z:0*
T0*0
_output_shapes
:����������l
IdentityIdentityRelu_1:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1 ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������: : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
3__inference_residual_unit_3_layer_call_fn_128022163

inputs#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�%
	unknown_4:��
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128018342x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::����������: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128017219

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_layer_call_fn_128021673

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128017066�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023068

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023192

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022820

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128017954

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023564

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022900

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128017442

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
9__inference_batch_normalization_6_layer_call_fn_128023099

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128017475�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�j
�
D__inference_model_layer_call_and_return_conditional_losses_128020513
input_1*
conv2d_128020336:@1
residual_unit_128020341:@@%
residual_unit_128020343:@%
residual_unit_128020345:@%
residual_unit_128020347:@%
residual_unit_128020349:@1
residual_unit_128020351:@@%
residual_unit_128020353:@%
residual_unit_128020355:@%
residual_unit_128020357:@%
residual_unit_128020359:@4
residual_unit_1_128020362:@�(
residual_unit_1_128020364:	�(
residual_unit_1_128020366:	�(
residual_unit_1_128020368:	�(
residual_unit_1_128020370:	�5
residual_unit_1_128020372:��(
residual_unit_1_128020374:	�(
residual_unit_1_128020376:	�(
residual_unit_1_128020378:	�(
residual_unit_1_128020380:	�4
residual_unit_1_128020382:@�(
residual_unit_1_128020384:	�(
residual_unit_1_128020386:	�(
residual_unit_1_128020388:	�(
residual_unit_1_128020390:	�5
residual_unit_2_128020393:��(
residual_unit_2_128020395:	�(
residual_unit_2_128020397:	�(
residual_unit_2_128020399:	�(
residual_unit_2_128020401:	�5
residual_unit_2_128020403:��(
residual_unit_2_128020405:	�(
residual_unit_2_128020407:	�(
residual_unit_2_128020409:	�(
residual_unit_2_128020411:	�5
residual_unit_3_128020414:��(
residual_unit_3_128020416:	�(
residual_unit_3_128020418:	�(
residual_unit_3_128020420:	�(
residual_unit_3_128020422:	�5
residual_unit_3_128020424:��(
residual_unit_3_128020426:	�(
residual_unit_3_128020428:	�(
residual_unit_3_128020430:	�(
residual_unit_3_128020432:	�5
residual_unit_3_128020434:��(
residual_unit_3_128020436:	�(
residual_unit_3_128020438:	�(
residual_unit_3_128020440:	�(
residual_unit_3_128020442:	�5
residual_unit_4_128020445:��(
residual_unit_4_128020447:	�(
residual_unit_4_128020449:	�(
residual_unit_4_128020451:	�(
residual_unit_4_128020453:	�5
residual_unit_4_128020455:��(
residual_unit_4_128020457:	�(
residual_unit_4_128020459:	�(
residual_unit_4_128020461:	�(
residual_unit_4_128020463:	�5
residual_unit_5_128020466:��(
residual_unit_5_128020468:	�(
residual_unit_5_128020470:	�(
residual_unit_5_128020472:	�(
residual_unit_5_128020474:	�5
residual_unit_5_128020476:��(
residual_unit_5_128020478:	�(
residual_unit_5_128020480:	�(
residual_unit_5_128020482:	�(
residual_unit_5_128020484:	�5
residual_unit_5_128020486:��(
residual_unit_5_128020488:	�(
residual_unit_5_128020490:	�(
residual_unit_5_128020492:	�(
residual_unit_5_128020494:	�#
dense_128020499:
��
dense_128020501:	�%
dense_1_128020504:
�� 
dense_1_128020506:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_128020336*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_128018056�
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_128018065�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128017066�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_128020341residual_unit_128020343residual_unit_128020345residual_unit_128020347residual_unit_128020349residual_unit_128020351residual_unit_128020353residual_unit_128020355residual_unit_128020357residual_unit_128020359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_residual_unit_layer_call_and_return_conditional_losses_128019429�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_128020362residual_unit_1_128020364residual_unit_1_128020366residual_unit_1_128020368residual_unit_1_128020370residual_unit_1_128020372residual_unit_1_128020374residual_unit_1_128020376residual_unit_1_128020378residual_unit_1_128020380residual_unit_1_128020382residual_unit_1_128020384residual_unit_1_128020386residual_unit_1_128020388residual_unit_1_128020390*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128019327�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_128020393residual_unit_2_128020395residual_unit_2_128020397residual_unit_2_128020399residual_unit_2_128020401residual_unit_2_128020403residual_unit_2_128020405residual_unit_2_128020407residual_unit_2_128020409residual_unit_2_128020411*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128019208�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_128020414residual_unit_3_128020416residual_unit_3_128020418residual_unit_3_128020420residual_unit_3_128020422residual_unit_3_128020424residual_unit_3_128020426residual_unit_3_128020428residual_unit_3_128020430residual_unit_3_128020432residual_unit_3_128020434residual_unit_3_128020436residual_unit_3_128020438residual_unit_3_128020440residual_unit_3_128020442*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128019106�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_128020445residual_unit_4_128020447residual_unit_4_128020449residual_unit_4_128020451residual_unit_4_128020453residual_unit_4_128020455residual_unit_4_128020457residual_unit_4_128020459residual_unit_4_128020461residual_unit_4_128020463*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128018987�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_128020466residual_unit_5_128020468residual_unit_5_128020470residual_unit_5_128020472residual_unit_5_128020474residual_unit_5_128020476residual_unit_5_128020478residual_unit_5_128020480residual_unit_5_128020482residual_unit_5_128020484residual_unit_5_128020486residual_unit_5_128020488residual_unit_5_128020490residual_unit_5_128020492residual_unit_5_128020494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128018885�
(global_average_pooling2d/PartitionedCallPartitionedCall0residual_unit_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128018039�
flatten/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_128018534�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_128020499dense_128020501*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_128018546�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_128020504dense_1_128020506*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_128018562�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_128018584v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^residual_unit/StatefulPartitionedCall(^residual_unit_1/StatefulPartitionedCall(^residual_unit_2/StatefulPartitionedCall(^residual_unit_3/StatefulPartitionedCall(^residual_unit_4/StatefulPartitionedCall(^residual_unit_5/StatefulPartitionedCall!^sampling/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%residual_unit/StatefulPartitionedCall%residual_unit/StatefulPartitionedCall2R
'residual_unit_1/StatefulPartitionedCall'residual_unit_1/StatefulPartitionedCall2R
'residual_unit_2/StatefulPartitionedCall'residual_unit_2/StatefulPartitionedCall2R
'residual_unit_3/StatefulPartitionedCall'residual_unit_3/StatefulPartitionedCall2R
'residual_unit_4/StatefulPartitionedCall'residual_unit_4/StatefulPartitionedCall2R
'residual_unit_5/StatefulPartitionedCall'residual_unit_5/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128017762

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_14_layer_call_fn_128023595

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128017987�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�&
�
)__inference_model_layer_call_fn_128021022

inputs!
unknown:@#
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@%

unknown_10:@�

unknown_11:	�

unknown_12:	�

unknown_13:	�

unknown_14:	�&

unknown_15:��

unknown_16:	�

unknown_17:	�

unknown_18:	�

unknown_19:	�%

unknown_20:@�

unknown_21:	�

unknown_22:	�

unknown_23:	�

unknown_24:	�&

unknown_25:��

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:	�&

unknown_30:��

unknown_31:	�

unknown_32:	�

unknown_33:	�

unknown_34:	�&

unknown_35:��

unknown_36:	�

unknown_37:	�

unknown_38:	�

unknown_39:	�&

unknown_40:��

unknown_41:	�

unknown_42:	�

unknown_43:	�

unknown_44:	�&

unknown_45:��

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	�&

unknown_50:��

unknown_51:	�

unknown_52:	�

unknown_53:	�

unknown_54:	�&

unknown_55:��

unknown_56:	�

unknown_57:	�

unknown_58:	�

unknown_59:	�&

unknown_60:��

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�&

unknown_65:��

unknown_66:	�

unknown_67:	�

unknown_68:	�

unknown_69:	�&

unknown_70:��

unknown_71:	�

unknown_72:	�

unknown_73:	�

unknown_74:	�

unknown_75:
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78*\
TinU
S2Q*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*T
_read_only_resource_inputs6
42	 !"%&'*+,/014569:;>?@CDEHIJMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_128019817p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_activation_layer_call_and_return_conditional_losses_128018065

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128017250

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023272

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
v
G__inference_sampling_layer_call_and_return_conditional_losses_128022714
inputs_0
inputs_1
identity�=
ShapeShapeinputs_1*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*(
_output_shapes
:����������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:����������}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:����������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
truedivRealDivinputs_1truediv/y:output:0*
T0*(
_output_shapes
:����������J
ExpExptruediv:z:0*
T0*(
_output_shapes
:����������Y
mulMulrandom_normal:z:0Exp:y:0*
T0*(
_output_shapes
:����������R
addAddV2mul:z:0inputs_0*
T0*(
_output_shapes
:����������P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023644

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_activation_layer_call_and_return_conditional_losses_128021668

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������@@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128017091

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023210

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������:
dense1
StatefulPartitionedCall:0����������<
dense_11
StatefulPartitionedCall:1����������=
sampling1
StatefulPartitionedCall:2����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
  _jit_compiled_convolution_op"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3main_layers
4skip_layers"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;main_layers
<skip_layers"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cmain_layers
Dskip_layers"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kmain_layers
Lskip_layers"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Smain_layers
Tskip_layers"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[main_layers
\skip_layers"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
o76
p77
w78
x79"
trackable_list_wrapper
�
0
1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
o46
p47
w48
x49"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_model_layer_call_fn_128018756
)__inference_model_layer_call_fn_128020853
)__inference_model_layer_call_fn_128021022
)__inference_model_layer_call_fn_128020153�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_model_layer_call_and_return_conditional_losses_128021333
D__inference_model_layer_call_and_return_conditional_losses_128021644
D__inference_model_layer_call_and_return_conditional_losses_128020333
D__inference_model_layer_call_and_return_conditional_losses_128020513�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
$__inference__wrapped_model_128017057input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_layer_call_fn_128021651�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_layer_call_and_return_conditional_losses_128021658�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%@2conv2d/kernel
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_activation_layer_call_fn_128021663�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_activation_layer_call_and_return_conditional_losses_128021668�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_layer_call_fn_128021673�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128021678�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
o
0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
O
0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_residual_unit_layer_call_fn_128021703
1__inference_residual_unit_layer_call_fn_128021728�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021769
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021810�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
@
�0
�1
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
h
�0
�1
�2
�3
�4
�5
�6
�7
�8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_residual_unit_1_layer_call_fn_128021845
3__inference_residual_unit_1_layer_call_fn_128021880�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021938
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021996�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
@
�0
�1
�3
�4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_residual_unit_2_layer_call_fn_128022021
3__inference_residual_unit_2_layer_call_fn_128022046�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022087
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022128�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
@
�0
�1
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
h
�0
�1
�2
�3
�4
�5
�6
�7
�8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_residual_unit_3_layer_call_fn_128022163
3__inference_residual_unit_3_layer_call_fn_128022198�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022256
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022314�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
@
�0
�1
�3
�4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_residual_unit_4_layer_call_fn_128022339
3__inference_residual_unit_4_layer_call_fn_128022364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022405
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022446�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
@
�0
�1
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
h
�0
�1
�2
�3
�4
�5
�6
�7
�8"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_residual_unit_5_layer_call_fn_128022481
3__inference_residual_unit_5_layer_call_fn_128022516�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022574
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022632�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
@
�0
�1
�3
�4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
<__inference_global_average_pooling2d_layer_call_fn_128022637�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128022643�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_layer_call_fn_128022648�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_layer_call_and_return_conditional_losses_128022654�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_layer_call_fn_128022663�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_layer_call_and_return_conditional_losses_128022673�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :
��2dense/kernel
:�2
dense/bias
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_1_layer_call_fn_128022682�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_1_layer_call_and_return_conditional_losses_128022692�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_sampling_layer_call_fn_128022698�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_sampling_layer_call_and_return_conditional_losses_128022714�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
7:5@@2residual_unit/conv2d_1/kernel
5:3@2'residual_unit/batch_normalization/gamma
4:2@2&residual_unit/batch_normalization/beta
7:5@@2residual_unit/conv2d_2/kernel
7:5@2)residual_unit/batch_normalization_1/gamma
6:4@2(residual_unit/batch_normalization_1/beta
=:;@ (2-residual_unit/batch_normalization/moving_mean
A:?@ (21residual_unit/batch_normalization/moving_variance
?:=@ (2/residual_unit/batch_normalization_1/moving_mean
C:A@ (23residual_unit/batch_normalization_1/moving_variance
::8@�2residual_unit_1/conv2d_3/kernel
::8�2+residual_unit_1/batch_normalization_2/gamma
9:7�2*residual_unit_1/batch_normalization_2/beta
;:9��2residual_unit_1/conv2d_4/kernel
::8�2+residual_unit_1/batch_normalization_3/gamma
9:7�2*residual_unit_1/batch_normalization_3/beta
::8@�2residual_unit_1/conv2d_5/kernel
::8�2+residual_unit_1/batch_normalization_4/gamma
9:7�2*residual_unit_1/batch_normalization_4/beta
B:@� (21residual_unit_1/batch_normalization_2/moving_mean
F:D� (25residual_unit_1/batch_normalization_2/moving_variance
B:@� (21residual_unit_1/batch_normalization_3/moving_mean
F:D� (25residual_unit_1/batch_normalization_3/moving_variance
B:@� (21residual_unit_1/batch_normalization_4/moving_mean
F:D� (25residual_unit_1/batch_normalization_4/moving_variance
;:9��2residual_unit_2/conv2d_6/kernel
::8�2+residual_unit_2/batch_normalization_5/gamma
9:7�2*residual_unit_2/batch_normalization_5/beta
;:9��2residual_unit_2/conv2d_7/kernel
::8�2+residual_unit_2/batch_normalization_6/gamma
9:7�2*residual_unit_2/batch_normalization_6/beta
B:@� (21residual_unit_2/batch_normalization_5/moving_mean
F:D� (25residual_unit_2/batch_normalization_5/moving_variance
B:@� (21residual_unit_2/batch_normalization_6/moving_mean
F:D� (25residual_unit_2/batch_normalization_6/moving_variance
;:9��2residual_unit_3/conv2d_8/kernel
::8�2+residual_unit_3/batch_normalization_7/gamma
9:7�2*residual_unit_3/batch_normalization_7/beta
;:9��2residual_unit_3/conv2d_9/kernel
::8�2+residual_unit_3/batch_normalization_8/gamma
9:7�2*residual_unit_3/batch_normalization_8/beta
<::��2 residual_unit_3/conv2d_10/kernel
::8�2+residual_unit_3/batch_normalization_9/gamma
9:7�2*residual_unit_3/batch_normalization_9/beta
B:@� (21residual_unit_3/batch_normalization_7/moving_mean
F:D� (25residual_unit_3/batch_normalization_7/moving_variance
B:@� (21residual_unit_3/batch_normalization_8/moving_mean
F:D� (25residual_unit_3/batch_normalization_8/moving_variance
B:@� (21residual_unit_3/batch_normalization_9/moving_mean
F:D� (25residual_unit_3/batch_normalization_9/moving_variance
<::��2 residual_unit_4/conv2d_11/kernel
;:9�2,residual_unit_4/batch_normalization_10/gamma
::8�2+residual_unit_4/batch_normalization_10/beta
<::��2 residual_unit_4/conv2d_12/kernel
;:9�2,residual_unit_4/batch_normalization_11/gamma
::8�2+residual_unit_4/batch_normalization_11/beta
C:A� (22residual_unit_4/batch_normalization_10/moving_mean
G:E� (26residual_unit_4/batch_normalization_10/moving_variance
C:A� (22residual_unit_4/batch_normalization_11/moving_mean
G:E� (26residual_unit_4/batch_normalization_11/moving_variance
<::��2 residual_unit_5/conv2d_13/kernel
;:9�2,residual_unit_5/batch_normalization_12/gamma
::8�2+residual_unit_5/batch_normalization_12/beta
<::��2 residual_unit_5/conv2d_14/kernel
;:9�2,residual_unit_5/batch_normalization_13/gamma
::8�2+residual_unit_5/batch_normalization_13/beta
<::��2 residual_unit_5/conv2d_15/kernel
;:9�2,residual_unit_5/batch_normalization_14/gamma
::8�2+residual_unit_5/batch_normalization_14/beta
C:A� (22residual_unit_5/batch_normalization_12/moving_mean
G:E� (26residual_unit_5/batch_normalization_12/moving_variance
C:A� (22residual_unit_5/batch_normalization_13/moving_mean
G:E� (26residual_unit_5/batch_normalization_13/moving_variance
C:A� (22residual_unit_5/batch_normalization_14/moving_mean
G:E� (26residual_unit_5/batch_normalization_14/moving_variance
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_layer_call_fn_128018756input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_layer_call_fn_128020853inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_layer_call_fn_128021022inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_layer_call_fn_128020153input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_layer_call_and_return_conditional_losses_128021333inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_layer_call_and_return_conditional_losses_128021644inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_layer_call_and_return_conditional_losses_128020333input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_layer_call_and_return_conditional_losses_128020513input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_128020684input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_layer_call_fn_128021651inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_layer_call_and_return_conditional_losses_128021658inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_activation_layer_call_fn_128021663inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_activation_layer_call_and_return_conditional_losses_128021668inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_layer_call_fn_128021673inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128021678inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_residual_unit_layer_call_fn_128021703inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
1__inference_residual_unit_layer_call_fn_128021728inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021769inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021810inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_residual_unit_1_layer_call_fn_128021845inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_residual_unit_1_layer_call_fn_128021880inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021938inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021996inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_residual_unit_2_layer_call_fn_128022021inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_residual_unit_2_layer_call_fn_128022046inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022087inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022128inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_residual_unit_3_layer_call_fn_128022163inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_residual_unit_3_layer_call_fn_128022198inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022256inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022314inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_residual_unit_4_layer_call_fn_128022339inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_residual_unit_4_layer_call_fn_128022364inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022405inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022446inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_residual_unit_5_layer_call_fn_128022481inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_residual_unit_5_layer_call_fn_128022516inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022574inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022632inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_global_average_pooling2d_layer_call_fn_128022637inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128022643inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_flatten_layer_call_fn_128022648inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_layer_call_and_return_conditional_losses_128022654inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_layer_call_fn_128022663inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_layer_call_and_return_conditional_losses_128022673inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_1_layer_call_fn_128022682inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_1_layer_call_and_return_conditional_losses_128022692inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sampling_layer_call_fn_128022698inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sampling_layer_call_and_return_conditional_losses_128022714inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_layer_call_fn_128022727
7__inference_batch_normalization_layer_call_fn_128022740�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022758
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022776�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_1_layer_call_fn_128022789
9__inference_batch_normalization_1_layer_call_fn_128022802�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022820
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022838�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_2_layer_call_fn_128022851
9__inference_batch_normalization_2_layer_call_fn_128022864�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022882
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022900�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_3_layer_call_fn_128022913
9__inference_batch_normalization_3_layer_call_fn_128022926�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022944
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022962�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_4_layer_call_fn_128022975
9__inference_batch_normalization_4_layer_call_fn_128022988�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023006
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023024�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_5_layer_call_fn_128023037
9__inference_batch_normalization_5_layer_call_fn_128023050�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023068
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023086�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_6_layer_call_fn_128023099
9__inference_batch_normalization_6_layer_call_fn_128023112�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023130
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023148�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_7_layer_call_fn_128023161
9__inference_batch_normalization_7_layer_call_fn_128023174�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023192
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023210�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_8_layer_call_fn_128023223
9__inference_batch_normalization_8_layer_call_fn_128023236�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023254
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023272�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_batch_normalization_9_layer_call_fn_128023285
9__inference_batch_normalization_9_layer_call_fn_128023298�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023316
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023334�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_10_layer_call_fn_128023347
:__inference_batch_normalization_10_layer_call_fn_128023360�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023378
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023396�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_11_layer_call_fn_128023409
:__inference_batch_normalization_11_layer_call_fn_128023422�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023440
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023458�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_12_layer_call_fn_128023471
:__inference_batch_normalization_12_layer_call_fn_128023484�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023502
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023520�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_13_layer_call_fn_128023533
:__inference_batch_normalization_13_layer_call_fn_128023546�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023564
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023582�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
:__inference_batch_normalization_14_layer_call_fn_128023595
:__inference_batch_normalization_14_layer_call_fn_128023608�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023626
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023644�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_layer_call_fn_128022727inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_layer_call_fn_128022740inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022758inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022776inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_1_layer_call_fn_128022789inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_1_layer_call_fn_128022802inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022820inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022838inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_2_layer_call_fn_128022851inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_2_layer_call_fn_128022864inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022882inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022900inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_3_layer_call_fn_128022913inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_3_layer_call_fn_128022926inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022944inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022962inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_4_layer_call_fn_128022975inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_4_layer_call_fn_128022988inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023006inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023024inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_5_layer_call_fn_128023037inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_5_layer_call_fn_128023050inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023068inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023086inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_6_layer_call_fn_128023099inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_6_layer_call_fn_128023112inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023130inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023148inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_7_layer_call_fn_128023161inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_7_layer_call_fn_128023174inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023192inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023210inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_8_layer_call_fn_128023223inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_8_layer_call_fn_128023236inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023254inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023272inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_batch_normalization_9_layer_call_fn_128023285inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_batch_normalization_9_layer_call_fn_128023298inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023316inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023334inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_10_layer_call_fn_128023347inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_10_layer_call_fn_128023360inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023378inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023396inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_11_layer_call_fn_128023409inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_11_layer_call_fn_128023422inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023440inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023458inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_12_layer_call_fn_128023471inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_12_layer_call_fn_128023484inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023502inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023520inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_13_layer_call_fn_128023533inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_13_layer_call_fn_128023546inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023564inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023582inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
:__inference_batch_normalization_14_layer_call_fn_128023595inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
:__inference_batch_normalization_14_layer_call_fn_128023608inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023626inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023644inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
$__inference__wrapped_model_128017057����������������������������������������������������������������������������opwx:�7
0�-
+�(
input_1�����������
� "���
)
dense �
dense����������
-
dense_1"�
dense_1����������
/
sampling#� 
sampling�����������
I__inference_activation_layer_call_and_return_conditional_losses_128021668h7�4
-�*
(�%
inputs���������@@@
� "-�*
#� 
0���������@@@
� �
.__inference_activation_layer_call_fn_128021663[7�4
-�*
(�%
inputs���������@@@
� " ����������@@@�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023378�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_128023396�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_10_layer_call_fn_128023347�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_10_layer_call_fn_128023360�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023440�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_128023458�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_11_layer_call_fn_128023409�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_11_layer_call_fn_128023422�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023502�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_128023520�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_12_layer_call_fn_128023471�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_12_layer_call_fn_128023484�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023564�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_128023582�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_13_layer_call_fn_128023533�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_13_layer_call_fn_128023546�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023626�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_128023644�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_14_layer_call_fn_128023595�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_14_layer_call_fn_128023608�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022820�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_128022838�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
9__inference_batch_normalization_1_layer_call_fn_128022789�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
9__inference_batch_normalization_1_layer_call_fn_128022802�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022882�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_128022900�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_2_layer_call_fn_128022851�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_2_layer_call_fn_128022864�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022944�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_128022962�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_3_layer_call_fn_128022913�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_3_layer_call_fn_128022926�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023006�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_128023024�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_4_layer_call_fn_128022975�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_4_layer_call_fn_128022988�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023068�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_128023086�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_5_layer_call_fn_128023037�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_5_layer_call_fn_128023050�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023130�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_128023148�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_6_layer_call_fn_128023099�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_6_layer_call_fn_128023112�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023192�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_128023210�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_7_layer_call_fn_128023161�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_7_layer_call_fn_128023174�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023254�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_128023272�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_8_layer_call_fn_128023223�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_8_layer_call_fn_128023236�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023316�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_128023334�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_9_layer_call_fn_128023285�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_9_layer_call_fn_128023298�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022758�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_layer_call_and_return_conditional_losses_128022776�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_layer_call_fn_128022727�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_layer_call_fn_128022740�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
E__inference_conv2d_layer_call_and_return_conditional_losses_128021658m9�6
/�,
*�'
inputs�����������
� "-�*
#� 
0���������@@@
� �
*__inference_conv2d_layer_call_fn_128021651`9�6
/�,
*�'
inputs�����������
� " ����������@@@�
F__inference_dense_1_layer_call_and_return_conditional_losses_128022692^wx0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_1_layer_call_fn_128022682Qwx0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_layer_call_and_return_conditional_losses_128022673^op0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_layer_call_fn_128022663Qop0�-
&�#
!�
inputs����������
� "������������
F__inference_flatten_layer_call_and_return_conditional_losses_128022654Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
+__inference_flatten_layer_call_fn_128022648M0�-
&�#
!�
inputs����������
� "������������
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_128022643�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
<__inference_global_average_pooling2d_layer_call_fn_128022637wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_128021678�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_layer_call_fn_128021673�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_model_layer_call_and_return_conditional_losses_128020333����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p 

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
D__inference_model_layer_call_and_return_conditional_losses_128020513����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
D__inference_model_layer_call_and_return_conditional_losses_128021333����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p 

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
D__inference_model_layer_call_and_return_conditional_losses_128021644����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
)__inference_model_layer_call_fn_128018756����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p 

 
� "]�Z
�
0����������
�
1����������
�
2�����������
)__inference_model_layer_call_fn_128020153����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p

 
� "]�Z
�
0����������
�
1����������
�
2�����������
)__inference_model_layer_call_fn_128020853����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p 

 
� "]�Z
�
0����������
�
1����������
�
2�����������
)__inference_model_layer_call_fn_128021022����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p

 
� "]�Z
�
0����������
�
1����������
�
2�����������
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021938����������������G�D
-�*
(�%
inputs���������  @
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_128021996����������������G�D
-�*
(�%
inputs���������  @
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_1_layer_call_fn_128021845����������������G�D
-�*
(�%
inputs���������  @
�

trainingp "!������������
3__inference_residual_unit_1_layer_call_fn_128021880����������������G�D
-�*
(�%
inputs���������  @
�

trainingp"!������������
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022087�����������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_128022128�����������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_2_layer_call_fn_128022021�����������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_2_layer_call_fn_128022046�����������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022256����������������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_128022314����������������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_3_layer_call_fn_128022163����������������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_3_layer_call_fn_128022198����������������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022405�����������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_128022446�����������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_4_layer_call_fn_128022339�����������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_4_layer_call_fn_128022364�����������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022574����������������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_128022632����������������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_5_layer_call_fn_128022481����������������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_5_layer_call_fn_128022516����������������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021769����������G�D
-�*
(�%
inputs���������  @
�

trainingp "-�*
#� 
0���������  @
� �
L__inference_residual_unit_layer_call_and_return_conditional_losses_128021810����������G�D
-�*
(�%
inputs���������  @
�

trainingp"-�*
#� 
0���������  @
� �
1__inference_residual_unit_layer_call_fn_128021703����������G�D
-�*
(�%
inputs���������  @
�

trainingp " ����������  @�
1__inference_residual_unit_layer_call_fn_128021728����������G�D
-�*
(�%
inputs���������  @
�

trainingp" ����������  @�
G__inference_sampling_layer_call_and_return_conditional_losses_128022714�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
,__inference_sampling_layer_call_fn_128022698y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
'__inference_signature_wrapper_128020684����������������������������������������������������������������������������opwxE�B
� 
;�8
6
input_1+�(
input_1�����������"���
)
dense �
dense����������
-
dense_1"�
dense_1����������
/
sampling#� 
sampling����������