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
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
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
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_137199912

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
"__inference__traced_save_137203137
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
%__inference__traced_restore_137203387��.
�
�
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202792

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
�
M
1__inference_max_pooling2d_layer_call_fn_137200901

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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137196294�
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
�G
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201802

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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137197023

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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202066

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
)__inference_dense_layer_call_fn_137201891

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137197774p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
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
��
�h
D__inference_model_layer_call_and_return_conditional_losses_137200872

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
��4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
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
��*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������V
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
:����������*
dtype0�
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*(
_output_shapes
:�����������
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*(
_output_shapes
:����������W
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
sampling/truedivRealDivdense_1/BiasAdd:output:0sampling/truediv/y:output:0*
T0*(
_output_shapes
:����������\
sampling/ExpExpsampling/truediv:z:0*
T0*(
_output_shapes
:����������t
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*(
_output_shapes
:����������r
sampling/addAddV2sampling/mul:z:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������j

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������b

Identity_2Identitysampling/add:z:0^NoOp*
T0*(
_output_shapes
:�����������2
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
�F
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137197570

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
�
s
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137197267

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
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137196990

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
�
X
<__inference_global_average_pooling2d_layer_call_fn_137201865

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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137197267i
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
�
�
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137196542

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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202420

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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137196606

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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202358

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
�?
�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137198436

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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202562

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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137196478

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
�
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137200906

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
9__inference_batch_normalization_8_layer_call_fn_137202451

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
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137196831�
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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137197151

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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137196798

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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202252

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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137201986

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
�?
�
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201356

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
�
s
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137201871

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
�
�
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202730

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
9__inference_batch_normalization_6_layer_call_fn_137202327

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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137196703�
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
�
e
I__inference_activation_layer_call_and_return_conditional_losses_137200896

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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202172

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
��
�@
%__inference__traced_restore_137203387
file_prefix8
assignvariableop_conv2d_kernel:@3
assignvariableop_1_dense_kernel:
��,
assignvariableop_2_dense_bias:	�5
!assignvariableop_3_dense_1_kernel:
��.
assignvariableop_4_dense_1_bias:	�J
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
9__inference_batch_normalization_6_layer_call_fn_137202340

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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137196734�
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
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202624

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
�j
�
D__inference_model_layer_call_and_return_conditional_losses_137199045

inputs*
conv2d_137198868:@1
residual_unit_137198873:@@%
residual_unit_137198875:@%
residual_unit_137198877:@%
residual_unit_137198879:@%
residual_unit_137198881:@1
residual_unit_137198883:@@%
residual_unit_137198885:@%
residual_unit_137198887:@%
residual_unit_137198889:@%
residual_unit_137198891:@4
residual_unit_1_137198894:@�(
residual_unit_1_137198896:	�(
residual_unit_1_137198898:	�(
residual_unit_1_137198900:	�(
residual_unit_1_137198902:	�5
residual_unit_1_137198904:��(
residual_unit_1_137198906:	�(
residual_unit_1_137198908:	�(
residual_unit_1_137198910:	�(
residual_unit_1_137198912:	�4
residual_unit_1_137198914:@�(
residual_unit_1_137198916:	�(
residual_unit_1_137198918:	�(
residual_unit_1_137198920:	�(
residual_unit_1_137198922:	�5
residual_unit_2_137198925:��(
residual_unit_2_137198927:	�(
residual_unit_2_137198929:	�(
residual_unit_2_137198931:	�(
residual_unit_2_137198933:	�5
residual_unit_2_137198935:��(
residual_unit_2_137198937:	�(
residual_unit_2_137198939:	�(
residual_unit_2_137198941:	�(
residual_unit_2_137198943:	�5
residual_unit_3_137198946:��(
residual_unit_3_137198948:	�(
residual_unit_3_137198950:	�(
residual_unit_3_137198952:	�(
residual_unit_3_137198954:	�5
residual_unit_3_137198956:��(
residual_unit_3_137198958:	�(
residual_unit_3_137198960:	�(
residual_unit_3_137198962:	�(
residual_unit_3_137198964:	�5
residual_unit_3_137198966:��(
residual_unit_3_137198968:	�(
residual_unit_3_137198970:	�(
residual_unit_3_137198972:	�(
residual_unit_3_137198974:	�5
residual_unit_4_137198977:��(
residual_unit_4_137198979:	�(
residual_unit_4_137198981:	�(
residual_unit_4_137198983:	�(
residual_unit_4_137198985:	�5
residual_unit_4_137198987:��(
residual_unit_4_137198989:	�(
residual_unit_4_137198991:	�(
residual_unit_4_137198993:	�(
residual_unit_4_137198995:	�5
residual_unit_5_137198998:��(
residual_unit_5_137199000:	�(
residual_unit_5_137199002:	�(
residual_unit_5_137199004:	�(
residual_unit_5_137199006:	�5
residual_unit_5_137199008:��(
residual_unit_5_137199010:	�(
residual_unit_5_137199012:	�(
residual_unit_5_137199014:	�(
residual_unit_5_137199016:	�5
residual_unit_5_137199018:��(
residual_unit_5_137199020:	�(
residual_unit_5_137199022:	�(
residual_unit_5_137199024:	�(
residual_unit_5_137199026:	�#
dense_137199031:
��
dense_137199033:	�%
dense_1_137199036:
�� 
dense_1_137199038:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_137198868*
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
E__inference_conv2d_layer_call_and_return_conditional_losses_137197284�
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
I__inference_activation_layer_call_and_return_conditional_losses_137197293�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137196294�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_137198873residual_unit_137198875residual_unit_137198877residual_unit_137198879residual_unit_137198881residual_unit_137198883residual_unit_137198885residual_unit_137198887residual_unit_137198889residual_unit_137198891*
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137198657�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_137198894residual_unit_1_137198896residual_unit_1_137198898residual_unit_1_137198900residual_unit_1_137198902residual_unit_1_137198904residual_unit_1_137198906residual_unit_1_137198908residual_unit_1_137198910residual_unit_1_137198912residual_unit_1_137198914residual_unit_1_137198916residual_unit_1_137198918residual_unit_1_137198920residual_unit_1_137198922*
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137198555�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_137198925residual_unit_2_137198927residual_unit_2_137198929residual_unit_2_137198931residual_unit_2_137198933residual_unit_2_137198935residual_unit_2_137198937residual_unit_2_137198939residual_unit_2_137198941residual_unit_2_137198943*
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137198436�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_137198946residual_unit_3_137198948residual_unit_3_137198950residual_unit_3_137198952residual_unit_3_137198954residual_unit_3_137198956residual_unit_3_137198958residual_unit_3_137198960residual_unit_3_137198962residual_unit_3_137198964residual_unit_3_137198966residual_unit_3_137198968residual_unit_3_137198970residual_unit_3_137198972residual_unit_3_137198974*
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137198334�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_137198977residual_unit_4_137198979residual_unit_4_137198981residual_unit_4_137198983residual_unit_4_137198985residual_unit_4_137198987residual_unit_4_137198989residual_unit_4_137198991residual_unit_4_137198993residual_unit_4_137198995*
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137198215�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_137198998residual_unit_5_137199000residual_unit_5_137199002residual_unit_5_137199004residual_unit_5_137199006residual_unit_5_137199008residual_unit_5_137199010residual_unit_5_137199012residual_unit_5_137199014residual_unit_5_137199016residual_unit_5_137199018residual_unit_5_137199020residual_unit_5_137199022residual_unit_5_137199024residual_unit_5_137199026*
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137198113�
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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137197267�
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
F__inference_flatten_layer_call_and_return_conditional_losses_137197762�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_137199031dense_137199033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137197774�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_137199036dense_1_137199038*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137197790�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_137197812v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
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
�&
�
)__inference_model_layer_call_fn_137200081

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
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
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
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_137197817p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
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
�
3__inference_residual_unit_5_layer_call_fn_137201744

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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137198113x
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
9__inference_batch_normalization_3_layer_call_fn_137202154

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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137196542�
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201315

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
:__inference_batch_normalization_13_layer_call_fn_137202774

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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137197182�
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
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_137197762

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
�0
�	
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137197490

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
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137196831

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
�[
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137198555

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
�
�
3__inference_residual_unit_1_layer_call_fn_137201108

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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137198555x
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
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202110

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
:__inference_batch_normalization_11_layer_call_fn_137202650

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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137197054�
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
�]
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201860

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
�
G
+__inference_flatten_layer_call_fn_137201876

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
F__inference_flatten_layer_call_and_return_conditional_losses_137197762a
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
�
�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202854

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
D__inference_dense_layer_call_and_return_conditional_losses_137197774

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202190

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
9__inference_batch_normalization_9_layer_call_fn_137202526

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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137196926�
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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137196926

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
�[
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201224

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
�
�
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137196670

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
�>
�

L__inference_residual_unit_layer_call_and_return_conditional_losses_137201038

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
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137196862

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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137197182

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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202872

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
I__inference_activation_layer_call_and_return_conditional_losses_137197293

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
�>
�

L__inference_residual_unit_layer_call_and_return_conditional_losses_137198657

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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137196639

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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137196511

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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137202004

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
�
�
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202376

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

�
3__inference_residual_unit_4_layer_call_fn_137201592

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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137198215x
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
�
�
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137197246

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
�
u
,__inference_sampling_layer_call_fn_137201926
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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_137197812p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202482

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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137196767

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
:__inference_batch_normalization_13_layer_call_fn_137202761

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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137197151�
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
9__inference_batch_normalization_5_layer_call_fn_137202278

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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137196670�
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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202438

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
9__inference_batch_normalization_8_layer_call_fn_137202464

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
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137196862�
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
�1
�

N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201633

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
�	
�
:__inference_batch_normalization_10_layer_call_fn_137202588

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
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137196990�
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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202810

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
�	
�
9__inference_batch_normalization_9_layer_call_fn_137202513

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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137196895�
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
9__inference_batch_normalization_2_layer_call_fn_137202092

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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137196478�
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137198334

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
�
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137196294

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
�
�
E__inference_conv2d_layer_call_and_return_conditional_losses_137200886

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
�
�
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137196414

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
��
�/
"__inference__traced_save_137203137
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
��:�:
��:�:@@:@:@:@@:@:@:@:@:@:@:@�:�:�:��:�:�:@�:�:�:�:�:�:�:�:�:��:�:�:��:�:�:�:�:�:�:��:�:�:��:�:�:��:�:�:�:�:�:�:�:�:��:�:�:��:�:�:�:�:�:�:��:�:�:��:�:�:��:�:�:�:�:�:�:�:�: 2(
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
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:,(
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
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202128

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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202234

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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137196575

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
:__inference_batch_normalization_14_layer_call_fn_137202823

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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137197215�
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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137197215

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
�/
�	
L__inference_residual_unit_layer_call_and_return_conditional_losses_137197337

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
�
:__inference_batch_normalization_11_layer_call_fn_137202637

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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137197023�
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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137197087

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
9__inference_batch_normalization_4_layer_call_fn_137202203

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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137196575�
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
�&
�
)__inference_model_layer_call_fn_137199381
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
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
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
<:����������:����������:����������*T
_read_only_resource_inputs6
42	 !"%&'*+,/014569:;>?@CDEHIJMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_137199045p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
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
9__inference_batch_normalization_5_layer_call_fn_137202265

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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137196639�
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
�1
�

N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137197643

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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202668

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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137200997

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
�&
�
'__inference_signature_wrapper_137199912
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
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
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
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__wrapped_model_137196285p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
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
t
G__inference_sampling_layer_call_and_return_conditional_losses_137197812

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
:����������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:����������}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:����������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
truedivRealDivinputs_1truediv/y:output:0*
T0*(
_output_shapes
:����������J
ExpExptruediv:z:0*
T0*(
_output_shapes
:����������Y
mulMulrandom_normal:z:0Exp:y:0*
T0*(
_output_shapes
:����������P
addAddV2mul:z:0inputs*
T0*(
_output_shapes
:����������P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_1_layer_call_fn_137201910

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137197790p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
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
�

�
3__inference_residual_unit_2_layer_call_fn_137201249

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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137197490x
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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137197054

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
�
3__inference_residual_unit_3_layer_call_fn_137201426

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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137198334x
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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202048

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
�

�
1__inference_residual_unit_layer_call_fn_137200931

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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137197337w
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
�j
�
D__inference_model_layer_call_and_return_conditional_losses_137197817

inputs*
conv2d_137197285:@1
residual_unit_137197338:@@%
residual_unit_137197340:@%
residual_unit_137197342:@%
residual_unit_137197344:@%
residual_unit_137197346:@1
residual_unit_137197348:@@%
residual_unit_137197350:@%
residual_unit_137197352:@%
residual_unit_137197354:@%
residual_unit_137197356:@4
residual_unit_1_137197418:@�(
residual_unit_1_137197420:	�(
residual_unit_1_137197422:	�(
residual_unit_1_137197424:	�(
residual_unit_1_137197426:	�5
residual_unit_1_137197428:��(
residual_unit_1_137197430:	�(
residual_unit_1_137197432:	�(
residual_unit_1_137197434:	�(
residual_unit_1_137197436:	�4
residual_unit_1_137197438:@�(
residual_unit_1_137197440:	�(
residual_unit_1_137197442:	�(
residual_unit_1_137197444:	�(
residual_unit_1_137197446:	�5
residual_unit_2_137197491:��(
residual_unit_2_137197493:	�(
residual_unit_2_137197495:	�(
residual_unit_2_137197497:	�(
residual_unit_2_137197499:	�5
residual_unit_2_137197501:��(
residual_unit_2_137197503:	�(
residual_unit_2_137197505:	�(
residual_unit_2_137197507:	�(
residual_unit_2_137197509:	�5
residual_unit_3_137197571:��(
residual_unit_3_137197573:	�(
residual_unit_3_137197575:	�(
residual_unit_3_137197577:	�(
residual_unit_3_137197579:	�5
residual_unit_3_137197581:��(
residual_unit_3_137197583:	�(
residual_unit_3_137197585:	�(
residual_unit_3_137197587:	�(
residual_unit_3_137197589:	�5
residual_unit_3_137197591:��(
residual_unit_3_137197593:	�(
residual_unit_3_137197595:	�(
residual_unit_3_137197597:	�(
residual_unit_3_137197599:	�5
residual_unit_4_137197644:��(
residual_unit_4_137197646:	�(
residual_unit_4_137197648:	�(
residual_unit_4_137197650:	�(
residual_unit_4_137197652:	�5
residual_unit_4_137197654:��(
residual_unit_4_137197656:	�(
residual_unit_4_137197658:	�(
residual_unit_4_137197660:	�(
residual_unit_4_137197662:	�5
residual_unit_5_137197724:��(
residual_unit_5_137197726:	�(
residual_unit_5_137197728:	�(
residual_unit_5_137197730:	�(
residual_unit_5_137197732:	�5
residual_unit_5_137197734:��(
residual_unit_5_137197736:	�(
residual_unit_5_137197738:	�(
residual_unit_5_137197740:	�(
residual_unit_5_137197742:	�5
residual_unit_5_137197744:��(
residual_unit_5_137197746:	�(
residual_unit_5_137197748:	�(
residual_unit_5_137197750:	�(
residual_unit_5_137197752:	�#
dense_137197775:
��
dense_137197777:	�%
dense_1_137197791:
�� 
dense_1_137197793:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_137197285*
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
E__inference_conv2d_layer_call_and_return_conditional_losses_137197284�
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
I__inference_activation_layer_call_and_return_conditional_losses_137197293�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137196294�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_137197338residual_unit_137197340residual_unit_137197342residual_unit_137197344residual_unit_137197346residual_unit_137197348residual_unit_137197350residual_unit_137197352residual_unit_137197354residual_unit_137197356*
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137197337�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_137197418residual_unit_1_137197420residual_unit_1_137197422residual_unit_1_137197424residual_unit_1_137197426residual_unit_1_137197428residual_unit_1_137197430residual_unit_1_137197432residual_unit_1_137197434residual_unit_1_137197436residual_unit_1_137197438residual_unit_1_137197440residual_unit_1_137197442residual_unit_1_137197444residual_unit_1_137197446*
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137197417�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_137197491residual_unit_2_137197493residual_unit_2_137197495residual_unit_2_137197497residual_unit_2_137197499residual_unit_2_137197501residual_unit_2_137197503residual_unit_2_137197505residual_unit_2_137197507residual_unit_2_137197509*
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137197490�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_137197571residual_unit_3_137197573residual_unit_3_137197575residual_unit_3_137197577residual_unit_3_137197579residual_unit_3_137197581residual_unit_3_137197583residual_unit_3_137197585residual_unit_3_137197587residual_unit_3_137197589residual_unit_3_137197591residual_unit_3_137197593residual_unit_3_137197595residual_unit_3_137197597residual_unit_3_137197599*
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137197570�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_137197644residual_unit_4_137197646residual_unit_4_137197648residual_unit_4_137197650residual_unit_4_137197652residual_unit_4_137197654residual_unit_4_137197656residual_unit_4_137197658residual_unit_4_137197660residual_unit_4_137197662*
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137197643�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_137197724residual_unit_5_137197726residual_unit_5_137197728residual_unit_5_137197730residual_unit_5_137197732residual_unit_5_137197734residual_unit_5_137197736residual_unit_5_137197738residual_unit_5_137197740residual_unit_5_137197742residual_unit_5_137197744residual_unit_5_137197746residual_unit_5_137197748residual_unit_5_137197750residual_unit_5_137197752*
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137197723�
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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137197267�
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
F__inference_flatten_layer_call_and_return_conditional_losses_137197762�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_137197775dense_137197777*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137197774�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_137197791dense_1_137197793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137197790�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_137197812v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
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
:__inference_batch_normalization_10_layer_call_fn_137202575

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
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137196959�
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
�F
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137197417

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
�
3__inference_residual_unit_1_layer_call_fn_137201073

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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137197417x
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

�
1__inference_residual_unit_layer_call_fn_137200956

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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137198657w
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
�j
�
D__inference_model_layer_call_and_return_conditional_losses_137199561
input_1*
conv2d_137199384:@1
residual_unit_137199389:@@%
residual_unit_137199391:@%
residual_unit_137199393:@%
residual_unit_137199395:@%
residual_unit_137199397:@1
residual_unit_137199399:@@%
residual_unit_137199401:@%
residual_unit_137199403:@%
residual_unit_137199405:@%
residual_unit_137199407:@4
residual_unit_1_137199410:@�(
residual_unit_1_137199412:	�(
residual_unit_1_137199414:	�(
residual_unit_1_137199416:	�(
residual_unit_1_137199418:	�5
residual_unit_1_137199420:��(
residual_unit_1_137199422:	�(
residual_unit_1_137199424:	�(
residual_unit_1_137199426:	�(
residual_unit_1_137199428:	�4
residual_unit_1_137199430:@�(
residual_unit_1_137199432:	�(
residual_unit_1_137199434:	�(
residual_unit_1_137199436:	�(
residual_unit_1_137199438:	�5
residual_unit_2_137199441:��(
residual_unit_2_137199443:	�(
residual_unit_2_137199445:	�(
residual_unit_2_137199447:	�(
residual_unit_2_137199449:	�5
residual_unit_2_137199451:��(
residual_unit_2_137199453:	�(
residual_unit_2_137199455:	�(
residual_unit_2_137199457:	�(
residual_unit_2_137199459:	�5
residual_unit_3_137199462:��(
residual_unit_3_137199464:	�(
residual_unit_3_137199466:	�(
residual_unit_3_137199468:	�(
residual_unit_3_137199470:	�5
residual_unit_3_137199472:��(
residual_unit_3_137199474:	�(
residual_unit_3_137199476:	�(
residual_unit_3_137199478:	�(
residual_unit_3_137199480:	�5
residual_unit_3_137199482:��(
residual_unit_3_137199484:	�(
residual_unit_3_137199486:	�(
residual_unit_3_137199488:	�(
residual_unit_3_137199490:	�5
residual_unit_4_137199493:��(
residual_unit_4_137199495:	�(
residual_unit_4_137199497:	�(
residual_unit_4_137199499:	�(
residual_unit_4_137199501:	�5
residual_unit_4_137199503:��(
residual_unit_4_137199505:	�(
residual_unit_4_137199507:	�(
residual_unit_4_137199509:	�(
residual_unit_4_137199511:	�5
residual_unit_5_137199514:��(
residual_unit_5_137199516:	�(
residual_unit_5_137199518:	�(
residual_unit_5_137199520:	�(
residual_unit_5_137199522:	�5
residual_unit_5_137199524:��(
residual_unit_5_137199526:	�(
residual_unit_5_137199528:	�(
residual_unit_5_137199530:	�(
residual_unit_5_137199532:	�5
residual_unit_5_137199534:��(
residual_unit_5_137199536:	�(
residual_unit_5_137199538:	�(
residual_unit_5_137199540:	�(
residual_unit_5_137199542:	�#
dense_137199547:
��
dense_137199549:	�%
dense_1_137199552:
�� 
dense_1_137199554:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_137199384*
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
E__inference_conv2d_layer_call_and_return_conditional_losses_137197284�
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
I__inference_activation_layer_call_and_return_conditional_losses_137197293�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137196294�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_137199389residual_unit_137199391residual_unit_137199393residual_unit_137199395residual_unit_137199397residual_unit_137199399residual_unit_137199401residual_unit_137199403residual_unit_137199405residual_unit_137199407*
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137197337�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_137199410residual_unit_1_137199412residual_unit_1_137199414residual_unit_1_137199416residual_unit_1_137199418residual_unit_1_137199420residual_unit_1_137199422residual_unit_1_137199424residual_unit_1_137199426residual_unit_1_137199428residual_unit_1_137199430residual_unit_1_137199432residual_unit_1_137199434residual_unit_1_137199436residual_unit_1_137199438*
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137197417�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_137199441residual_unit_2_137199443residual_unit_2_137199445residual_unit_2_137199447residual_unit_2_137199449residual_unit_2_137199451residual_unit_2_137199453residual_unit_2_137199455residual_unit_2_137199457residual_unit_2_137199459*
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137197490�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_137199462residual_unit_3_137199464residual_unit_3_137199466residual_unit_3_137199468residual_unit_3_137199470residual_unit_3_137199472residual_unit_3_137199474residual_unit_3_137199476residual_unit_3_137199478residual_unit_3_137199480residual_unit_3_137199482residual_unit_3_137199484residual_unit_3_137199486residual_unit_3_137199488residual_unit_3_137199490*
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137197570�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_137199493residual_unit_4_137199495residual_unit_4_137199497residual_unit_4_137199499residual_unit_4_137199501residual_unit_4_137199503residual_unit_4_137199505residual_unit_4_137199507residual_unit_4_137199509residual_unit_4_137199511*
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137197643�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_137199514residual_unit_5_137199516residual_unit_5_137199518residual_unit_5_137199520residual_unit_5_137199522residual_unit_5_137199524residual_unit_5_137199526residual_unit_5_137199528residual_unit_5_137199530residual_unit_5_137199532residual_unit_5_137199534residual_unit_5_137199536residual_unit_5_137199538residual_unit_5_137199540residual_unit_5_137199542*
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137197723�
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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137197267�
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
F__inference_flatten_layer_call_and_return_conditional_losses_137197762�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_137199547dense_137199549*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137197774�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_137199552dense_1_137199554*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137197790�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_137197812v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
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
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_137201882

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
�
�
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137196350

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
9__inference_batch_normalization_4_layer_call_fn_137202216

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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137196606�
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
v
G__inference_sampling_layer_call_and_return_conditional_losses_137201942
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
:����������*
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:����������}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:����������N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
truedivRealDivinputs_1truediv/y:output:0*
T0*(
_output_shapes
:����������J
ExpExptruediv:z:0*
T0*(
_output_shapes
:����������Y
mulMulrandom_normal:z:0Exp:y:0*
T0*(
_output_shapes
:����������R
addAddV2mul:z:0inputs_0*
T0*(
_output_shapes
:����������P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�	
�
:__inference_batch_normalization_12_layer_call_fn_137202712

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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137197118�
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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137196319

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
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137196959

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
9__inference_batch_normalization_7_layer_call_fn_137202389

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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137196767�
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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202314

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
F__inference_dense_1_layer_call_and_return_conditional_losses_137201920

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
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
�
3__inference_residual_unit_5_layer_call_fn_137201709

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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137197723x
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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137196383

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
�
�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202606

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
��
�b
$__inference__wrapped_model_137196285
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
��:
+model_dense_biasadd_readvariableop_resource:	�@
,model_dense_1_matmul_readvariableop_resource:
��<
-model_dense_1_biasadd_readvariableop_resource:	�
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
��*
dtype0�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������b
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
:����������*
dtype0�
 model/sampling/random_normal/mulMul:model/sampling/random_normal/RandomStandardNormal:output:0,model/sampling/random_normal/stddev:output:0*
T0*(
_output_shapes
:�����������
model/sampling/random_normalAddV2$model/sampling/random_normal/mul:z:0*model/sampling/random_normal/mean:output:0*
T0*(
_output_shapes
:����������]
model/sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
model/sampling/truedivRealDivmodel/dense_1/BiasAdd:output:0!model/sampling/truediv/y:output:0*
T0*(
_output_shapes
:����������h
model/sampling/ExpExpmodel/sampling/truediv:z:0*
T0*(
_output_shapes
:�����������
model/sampling/mulMul model/sampling/random_normal:z:0model/sampling/Exp:y:0*
T0*(
_output_shapes
:�����������
model/sampling/addAddV2model/sampling/mul:z:0model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������l
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������p

Identity_1Identitymodel/dense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������h

Identity_2Identitymodel/sampling/add:z:0^NoOp*
T0*(
_output_shapes
:�����������)
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
�
�
3__inference_residual_unit_3_layer_call_fn_137201391

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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137197570x
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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137196703

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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137196447

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

�
3__inference_residual_unit_4_layer_call_fn_137201567

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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137197643x
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
�
:__inference_batch_normalization_12_layer_call_fn_137202699

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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137197087�
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
9__inference_batch_normalization_3_layer_call_fn_137202141

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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137196511�
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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202748

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
�]
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137198113

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
�F
�
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201166

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
�
�
E__inference_conv2d_layer_call_and_return_conditional_losses_137197284

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
�&
�
)__inference_model_layer_call_fn_137197984
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
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
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
<:����������:����������:����������*r
_read_only_resource_inputsT
RP	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_137197817p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
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
�[
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201542

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
�[
D__inference_model_layer_call_and_return_conditional_losses_137200561

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
��4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�
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
��*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������V
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
:����������*
dtype0�
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*(
_output_shapes
:�����������
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*(
_output_shapes
:����������W
sampling/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
sampling/truedivRealDivdense_1/BiasAdd:output:0sampling/truediv/y:output:0*
T0*(
_output_shapes
:����������\
sampling/ExpExpsampling/truediv:z:0*
T0*(
_output_shapes
:����������t
sampling/mulMulsampling/random_normal:z:0sampling/Exp:y:0*
T0*(
_output_shapes
:����������r
sampling/addAddV2sampling/mul:z:0dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������j

Identity_1Identitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������b

Identity_2Identitysampling/add:z:0^NoOp*
T0*(
_output_shapes
:�����������%
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
�	
�
F__inference_dense_1_layer_call_and_return_conditional_losses_137197790

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
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
�
9__inference_batch_normalization_1_layer_call_fn_137202030

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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137196414�
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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137196895

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
�&
�
)__inference_model_layer_call_fn_137200250

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
��

unknown_76:	�

unknown_77:
��

unknown_78:	�
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
<:����������:����������:����������*T
_read_only_resource_inputs6
42	 !"%&'*+,/014569:;>?@CDEHIJMNOP*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_137199045p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
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
9__inference_batch_normalization_1_layer_call_fn_137202017

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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137196383�
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
9__inference_batch_normalization_7_layer_call_fn_137202402

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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137196798�
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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137197118

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
�	
�
:__inference_batch_normalization_14_layer_call_fn_137202836

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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137197246�
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
�@
�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137198215

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
�
�
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202500

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
3__inference_residual_unit_2_layer_call_fn_137201274

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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137198436x
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
�j
�
D__inference_model_layer_call_and_return_conditional_losses_137199741
input_1*
conv2d_137199564:@1
residual_unit_137199569:@@%
residual_unit_137199571:@%
residual_unit_137199573:@%
residual_unit_137199575:@%
residual_unit_137199577:@1
residual_unit_137199579:@@%
residual_unit_137199581:@%
residual_unit_137199583:@%
residual_unit_137199585:@%
residual_unit_137199587:@4
residual_unit_1_137199590:@�(
residual_unit_1_137199592:	�(
residual_unit_1_137199594:	�(
residual_unit_1_137199596:	�(
residual_unit_1_137199598:	�5
residual_unit_1_137199600:��(
residual_unit_1_137199602:	�(
residual_unit_1_137199604:	�(
residual_unit_1_137199606:	�(
residual_unit_1_137199608:	�4
residual_unit_1_137199610:@�(
residual_unit_1_137199612:	�(
residual_unit_1_137199614:	�(
residual_unit_1_137199616:	�(
residual_unit_1_137199618:	�5
residual_unit_2_137199621:��(
residual_unit_2_137199623:	�(
residual_unit_2_137199625:	�(
residual_unit_2_137199627:	�(
residual_unit_2_137199629:	�5
residual_unit_2_137199631:��(
residual_unit_2_137199633:	�(
residual_unit_2_137199635:	�(
residual_unit_2_137199637:	�(
residual_unit_2_137199639:	�5
residual_unit_3_137199642:��(
residual_unit_3_137199644:	�(
residual_unit_3_137199646:	�(
residual_unit_3_137199648:	�(
residual_unit_3_137199650:	�5
residual_unit_3_137199652:��(
residual_unit_3_137199654:	�(
residual_unit_3_137199656:	�(
residual_unit_3_137199658:	�(
residual_unit_3_137199660:	�5
residual_unit_3_137199662:��(
residual_unit_3_137199664:	�(
residual_unit_3_137199666:	�(
residual_unit_3_137199668:	�(
residual_unit_3_137199670:	�5
residual_unit_4_137199673:��(
residual_unit_4_137199675:	�(
residual_unit_4_137199677:	�(
residual_unit_4_137199679:	�(
residual_unit_4_137199681:	�5
residual_unit_4_137199683:��(
residual_unit_4_137199685:	�(
residual_unit_4_137199687:	�(
residual_unit_4_137199689:	�(
residual_unit_4_137199691:	�5
residual_unit_5_137199694:��(
residual_unit_5_137199696:	�(
residual_unit_5_137199698:	�(
residual_unit_5_137199700:	�(
residual_unit_5_137199702:	�5
residual_unit_5_137199704:��(
residual_unit_5_137199706:	�(
residual_unit_5_137199708:	�(
residual_unit_5_137199710:	�(
residual_unit_5_137199712:	�5
residual_unit_5_137199714:��(
residual_unit_5_137199716:	�(
residual_unit_5_137199718:	�(
residual_unit_5_137199720:	�(
residual_unit_5_137199722:	�#
dense_137199727:
��
dense_137199729:	�%
dense_1_137199732:
�� 
dense_1_137199734:	�
identity

identity_1

identity_2��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�%residual_unit/StatefulPartitionedCall�'residual_unit_1/StatefulPartitionedCall�'residual_unit_2/StatefulPartitionedCall�'residual_unit_3/StatefulPartitionedCall�'residual_unit_4/StatefulPartitionedCall�'residual_unit_5/StatefulPartitionedCall� sampling/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_137199564*
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
E__inference_conv2d_layer_call_and_return_conditional_losses_137197284�
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
I__inference_activation_layer_call_and_return_conditional_losses_137197293�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137196294�
%residual_unit/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0residual_unit_137199569residual_unit_137199571residual_unit_137199573residual_unit_137199575residual_unit_137199577residual_unit_137199579residual_unit_137199581residual_unit_137199583residual_unit_137199585residual_unit_137199587*
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137198657�
'residual_unit_1/StatefulPartitionedCallStatefulPartitionedCall.residual_unit/StatefulPartitionedCall:output:0residual_unit_1_137199590residual_unit_1_137199592residual_unit_1_137199594residual_unit_1_137199596residual_unit_1_137199598residual_unit_1_137199600residual_unit_1_137199602residual_unit_1_137199604residual_unit_1_137199606residual_unit_1_137199608residual_unit_1_137199610residual_unit_1_137199612residual_unit_1_137199614residual_unit_1_137199616residual_unit_1_137199618*
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137198555�
'residual_unit_2/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_1/StatefulPartitionedCall:output:0residual_unit_2_137199621residual_unit_2_137199623residual_unit_2_137199625residual_unit_2_137199627residual_unit_2_137199629residual_unit_2_137199631residual_unit_2_137199633residual_unit_2_137199635residual_unit_2_137199637residual_unit_2_137199639*
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137198436�
'residual_unit_3/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_2/StatefulPartitionedCall:output:0residual_unit_3_137199642residual_unit_3_137199644residual_unit_3_137199646residual_unit_3_137199648residual_unit_3_137199650residual_unit_3_137199652residual_unit_3_137199654residual_unit_3_137199656residual_unit_3_137199658residual_unit_3_137199660residual_unit_3_137199662residual_unit_3_137199664residual_unit_3_137199666residual_unit_3_137199668residual_unit_3_137199670*
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137198334�
'residual_unit_4/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_3/StatefulPartitionedCall:output:0residual_unit_4_137199673residual_unit_4_137199675residual_unit_4_137199677residual_unit_4_137199679residual_unit_4_137199681residual_unit_4_137199683residual_unit_4_137199685residual_unit_4_137199687residual_unit_4_137199689residual_unit_4_137199691*
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137198215�
'residual_unit_5/StatefulPartitionedCallStatefulPartitionedCall0residual_unit_4/StatefulPartitionedCall:output:0residual_unit_5_137199694residual_unit_5_137199696residual_unit_5_137199698residual_unit_5_137199700residual_unit_5_137199702residual_unit_5_137199704residual_unit_5_137199706residual_unit_5_137199708residual_unit_5_137199710residual_unit_5_137199712residual_unit_5_137199714residual_unit_5_137199716residual_unit_5_137199718residual_unit_5_137199720residual_unit_5_137199722*
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137198113�
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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137197267�
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
F__inference_flatten_layer_call_and_return_conditional_losses_137197762�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_137199727dense_137199729*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_137197774�
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_137199732dense_1_137199734*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_137197790�
 sampling/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
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
GPU2*0J 8� *P
fKRI
G__inference_sampling_layer_call_and_return_conditional_losses_137197812v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������{

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
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
�	
�
7__inference_batch_normalization_layer_call_fn_137201955

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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137196319�
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
�F
�
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201484

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
�
�
*__inference_conv2d_layer_call_fn_137200879

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
E__inference_conv2d_layer_call_and_return_conditional_losses_137197284w
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
�G
�
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137197723

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
�
J
.__inference_activation_layer_call_fn_137200891

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
I__inference_activation_layer_call_and_return_conditional_losses_137197293h
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
�	
�
9__inference_batch_normalization_2_layer_call_fn_137202079

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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137196447�
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
D__inference_dense_layer_call_and_return_conditional_losses_137201901

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202686

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
�
�
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202544

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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137196734

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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202296

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
7__inference_batch_normalization_layer_call_fn_137201968

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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137196350�
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
�@
�
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201674

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
StatefulPartitionedCall:0����������<
dense_11
StatefulPartitionedCall:1����������=
sampling1
StatefulPartitionedCall:2����������tensorflow/serving/predict:��
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
)__inference_model_layer_call_fn_137197984
)__inference_model_layer_call_fn_137200081
)__inference_model_layer_call_fn_137200250
)__inference_model_layer_call_fn_137199381�
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
D__inference_model_layer_call_and_return_conditional_losses_137200561
D__inference_model_layer_call_and_return_conditional_losses_137200872
D__inference_model_layer_call_and_return_conditional_losses_137199561
D__inference_model_layer_call_and_return_conditional_losses_137199741�
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
$__inference__wrapped_model_137196285input_1"�
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
*__inference_conv2d_layer_call_fn_137200879�
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
E__inference_conv2d_layer_call_and_return_conditional_losses_137200886�
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
.__inference_activation_layer_call_fn_137200891�
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
I__inference_activation_layer_call_and_return_conditional_losses_137200896�
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
1__inference_max_pooling2d_layer_call_fn_137200901�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137200906�
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
1__inference_residual_unit_layer_call_fn_137200931
1__inference_residual_unit_layer_call_fn_137200956�
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137200997
L__inference_residual_unit_layer_call_and_return_conditional_losses_137201038�
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
3__inference_residual_unit_1_layer_call_fn_137201073
3__inference_residual_unit_1_layer_call_fn_137201108�
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201166
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201224�
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
3__inference_residual_unit_2_layer_call_fn_137201249
3__inference_residual_unit_2_layer_call_fn_137201274�
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201315
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201356�
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
3__inference_residual_unit_3_layer_call_fn_137201391
3__inference_residual_unit_3_layer_call_fn_137201426�
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201484
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201542�
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
3__inference_residual_unit_4_layer_call_fn_137201567
3__inference_residual_unit_4_layer_call_fn_137201592�
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201633
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201674�
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
3__inference_residual_unit_5_layer_call_fn_137201709
3__inference_residual_unit_5_layer_call_fn_137201744�
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201802
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201860�
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
<__inference_global_average_pooling2d_layer_call_fn_137201865�
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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137201871�
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
+__inference_flatten_layer_call_fn_137201876�
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
F__inference_flatten_layer_call_and_return_conditional_losses_137201882�
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
)__inference_dense_layer_call_fn_137201891�
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
D__inference_dense_layer_call_and_return_conditional_losses_137201901�
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
��2dense/kernel
:�2
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
+__inference_dense_1_layer_call_fn_137201910�
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
F__inference_dense_1_layer_call_and_return_conditional_losses_137201920�
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
��2dense_1/kernel
:�2dense_1/bias
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
,__inference_sampling_layer_call_fn_137201926�
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
G__inference_sampling_layer_call_and_return_conditional_losses_137201942�
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
)__inference_model_layer_call_fn_137197984input_1"�
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
)__inference_model_layer_call_fn_137200081inputs"�
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
)__inference_model_layer_call_fn_137200250inputs"�
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
)__inference_model_layer_call_fn_137199381input_1"�
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
D__inference_model_layer_call_and_return_conditional_losses_137200561inputs"�
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
D__inference_model_layer_call_and_return_conditional_losses_137200872inputs"�
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
D__inference_model_layer_call_and_return_conditional_losses_137199561input_1"�
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
D__inference_model_layer_call_and_return_conditional_losses_137199741input_1"�
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
'__inference_signature_wrapper_137199912input_1"�
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
*__inference_conv2d_layer_call_fn_137200879inputs"�
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
E__inference_conv2d_layer_call_and_return_conditional_losses_137200886inputs"�
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
.__inference_activation_layer_call_fn_137200891inputs"�
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
I__inference_activation_layer_call_and_return_conditional_losses_137200896inputs"�
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
1__inference_max_pooling2d_layer_call_fn_137200901inputs"�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137200906inputs"�
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
1__inference_residual_unit_layer_call_fn_137200931inputs"�
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
1__inference_residual_unit_layer_call_fn_137200956inputs"�
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137200997inputs"�
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
L__inference_residual_unit_layer_call_and_return_conditional_losses_137201038inputs"�
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
3__inference_residual_unit_1_layer_call_fn_137201073inputs"�
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
3__inference_residual_unit_1_layer_call_fn_137201108inputs"�
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201166inputs"�
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
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201224inputs"�
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
3__inference_residual_unit_2_layer_call_fn_137201249inputs"�
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
3__inference_residual_unit_2_layer_call_fn_137201274inputs"�
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201315inputs"�
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
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201356inputs"�
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
3__inference_residual_unit_3_layer_call_fn_137201391inputs"�
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
3__inference_residual_unit_3_layer_call_fn_137201426inputs"�
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201484inputs"�
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
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201542inputs"�
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
3__inference_residual_unit_4_layer_call_fn_137201567inputs"�
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
3__inference_residual_unit_4_layer_call_fn_137201592inputs"�
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201633inputs"�
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
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201674inputs"�
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
3__inference_residual_unit_5_layer_call_fn_137201709inputs"�
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
3__inference_residual_unit_5_layer_call_fn_137201744inputs"�
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201802inputs"�
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
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201860inputs"�
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
<__inference_global_average_pooling2d_layer_call_fn_137201865inputs"�
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
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137201871inputs"�
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
+__inference_flatten_layer_call_fn_137201876inputs"�
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
F__inference_flatten_layer_call_and_return_conditional_losses_137201882inputs"�
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
)__inference_dense_layer_call_fn_137201891inputs"�
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
D__inference_dense_layer_call_and_return_conditional_losses_137201901inputs"�
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
+__inference_dense_1_layer_call_fn_137201910inputs"�
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
F__inference_dense_1_layer_call_and_return_conditional_losses_137201920inputs"�
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
,__inference_sampling_layer_call_fn_137201926inputs/0inputs/1"�
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
G__inference_sampling_layer_call_and_return_conditional_losses_137201942inputs/0inputs/1"�
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
7__inference_batch_normalization_layer_call_fn_137201955
7__inference_batch_normalization_layer_call_fn_137201968�
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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137201986
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137202004�
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
9__inference_batch_normalization_1_layer_call_fn_137202017
9__inference_batch_normalization_1_layer_call_fn_137202030�
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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202048
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202066�
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
9__inference_batch_normalization_2_layer_call_fn_137202079
9__inference_batch_normalization_2_layer_call_fn_137202092�
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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202110
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202128�
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
9__inference_batch_normalization_3_layer_call_fn_137202141
9__inference_batch_normalization_3_layer_call_fn_137202154�
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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202172
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202190�
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
9__inference_batch_normalization_4_layer_call_fn_137202203
9__inference_batch_normalization_4_layer_call_fn_137202216�
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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202234
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202252�
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
9__inference_batch_normalization_5_layer_call_fn_137202265
9__inference_batch_normalization_5_layer_call_fn_137202278�
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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202296
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202314�
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
9__inference_batch_normalization_6_layer_call_fn_137202327
9__inference_batch_normalization_6_layer_call_fn_137202340�
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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202358
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202376�
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
9__inference_batch_normalization_7_layer_call_fn_137202389
9__inference_batch_normalization_7_layer_call_fn_137202402�
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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202420
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202438�
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
9__inference_batch_normalization_8_layer_call_fn_137202451
9__inference_batch_normalization_8_layer_call_fn_137202464�
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
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202482
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202500�
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
9__inference_batch_normalization_9_layer_call_fn_137202513
9__inference_batch_normalization_9_layer_call_fn_137202526�
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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202544
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202562�
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
:__inference_batch_normalization_10_layer_call_fn_137202575
:__inference_batch_normalization_10_layer_call_fn_137202588�
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
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202606
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202624�
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
:__inference_batch_normalization_11_layer_call_fn_137202637
:__inference_batch_normalization_11_layer_call_fn_137202650�
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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202668
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202686�
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
:__inference_batch_normalization_12_layer_call_fn_137202699
:__inference_batch_normalization_12_layer_call_fn_137202712�
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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202730
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202748�
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
:__inference_batch_normalization_13_layer_call_fn_137202761
:__inference_batch_normalization_13_layer_call_fn_137202774�
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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202792
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202810�
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
:__inference_batch_normalization_14_layer_call_fn_137202823
:__inference_batch_normalization_14_layer_call_fn_137202836�
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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202854
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202872�
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
7__inference_batch_normalization_layer_call_fn_137201955inputs"�
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
7__inference_batch_normalization_layer_call_fn_137201968inputs"�
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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137201986inputs"�
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
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137202004inputs"�
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
9__inference_batch_normalization_1_layer_call_fn_137202017inputs"�
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
9__inference_batch_normalization_1_layer_call_fn_137202030inputs"�
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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202048inputs"�
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
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202066inputs"�
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
9__inference_batch_normalization_2_layer_call_fn_137202079inputs"�
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
9__inference_batch_normalization_2_layer_call_fn_137202092inputs"�
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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202110inputs"�
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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202128inputs"�
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
9__inference_batch_normalization_3_layer_call_fn_137202141inputs"�
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
9__inference_batch_normalization_3_layer_call_fn_137202154inputs"�
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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202172inputs"�
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
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202190inputs"�
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
9__inference_batch_normalization_4_layer_call_fn_137202203inputs"�
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
9__inference_batch_normalization_4_layer_call_fn_137202216inputs"�
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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202234inputs"�
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
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202252inputs"�
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
9__inference_batch_normalization_5_layer_call_fn_137202265inputs"�
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
9__inference_batch_normalization_5_layer_call_fn_137202278inputs"�
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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202296inputs"�
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
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202314inputs"�
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
9__inference_batch_normalization_6_layer_call_fn_137202327inputs"�
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
9__inference_batch_normalization_6_layer_call_fn_137202340inputs"�
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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202358inputs"�
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
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202376inputs"�
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
9__inference_batch_normalization_7_layer_call_fn_137202389inputs"�
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
9__inference_batch_normalization_7_layer_call_fn_137202402inputs"�
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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202420inputs"�
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
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202438inputs"�
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
9__inference_batch_normalization_8_layer_call_fn_137202451inputs"�
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
9__inference_batch_normalization_8_layer_call_fn_137202464inputs"�
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
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202482inputs"�
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
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202500inputs"�
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
9__inference_batch_normalization_9_layer_call_fn_137202513inputs"�
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
9__inference_batch_normalization_9_layer_call_fn_137202526inputs"�
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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202544inputs"�
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
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202562inputs"�
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
:__inference_batch_normalization_10_layer_call_fn_137202575inputs"�
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
:__inference_batch_normalization_10_layer_call_fn_137202588inputs"�
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
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202606inputs"�
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
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202624inputs"�
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
:__inference_batch_normalization_11_layer_call_fn_137202637inputs"�
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
:__inference_batch_normalization_11_layer_call_fn_137202650inputs"�
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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202668inputs"�
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
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202686inputs"�
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
:__inference_batch_normalization_12_layer_call_fn_137202699inputs"�
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
:__inference_batch_normalization_12_layer_call_fn_137202712inputs"�
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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202730inputs"�
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
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202748inputs"�
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
:__inference_batch_normalization_13_layer_call_fn_137202761inputs"�
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
:__inference_batch_normalization_13_layer_call_fn_137202774inputs"�
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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202792inputs"�
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
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202810inputs"�
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
:__inference_batch_normalization_14_layer_call_fn_137202823inputs"�
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
:__inference_batch_normalization_14_layer_call_fn_137202836inputs"�
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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202854inputs"�
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
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202872inputs"�
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
$__inference__wrapped_model_137196285����������������������������������������������������������������������������opwx:�7
0�-
+�(
input_1�����������
� "���
)
dense �
dense����������
-
dense_1"�
dense_1����������
/
sampling#� 
sampling�����������
I__inference_activation_layer_call_and_return_conditional_losses_137200896h7�4
-�*
(�%
inputs���������@@@
� "-�*
#� 
0���������@@@
� �
.__inference_activation_layer_call_fn_137200891[7�4
-�*
(�%
inputs���������@@@
� " ����������@@@�
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202606�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_10_layer_call_and_return_conditional_losses_137202624�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_10_layer_call_fn_137202575�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_10_layer_call_fn_137202588�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202668�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_11_layer_call_and_return_conditional_losses_137202686�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_11_layer_call_fn_137202637�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_11_layer_call_fn_137202650�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202730�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_12_layer_call_and_return_conditional_losses_137202748�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_12_layer_call_fn_137202699�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_12_layer_call_fn_137202712�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202792�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_13_layer_call_and_return_conditional_losses_137202810�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_13_layer_call_fn_137202761�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_13_layer_call_fn_137202774�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202854�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
U__inference_batch_normalization_14_layer_call_and_return_conditional_losses_137202872�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
:__inference_batch_normalization_14_layer_call_fn_137202823�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
:__inference_batch_normalization_14_layer_call_fn_137202836�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202048�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_137202066�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
9__inference_batch_normalization_1_layer_call_fn_137202017�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
9__inference_batch_normalization_1_layer_call_fn_137202030�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202110�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_137202128�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_2_layer_call_fn_137202079�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_2_layer_call_fn_137202092�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202172�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_137202190�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_3_layer_call_fn_137202141�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_3_layer_call_fn_137202154�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202234�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_137202252�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_4_layer_call_fn_137202203�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_4_layer_call_fn_137202216�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202296�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_137202314�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_5_layer_call_fn_137202265�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_5_layer_call_fn_137202278�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202358�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_137202376�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_6_layer_call_fn_137202327�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_6_layer_call_fn_137202340�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202420�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_137202438�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_7_layer_call_fn_137202389�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_7_layer_call_fn_137202402�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202482�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_137202500�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_8_layer_call_fn_137202451�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_8_layer_call_fn_137202464�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202544�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
T__inference_batch_normalization_9_layer_call_and_return_conditional_losses_137202562�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
9__inference_batch_normalization_9_layer_call_fn_137202513�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
9__inference_batch_normalization_9_layer_call_fn_137202526�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137201986�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_layer_call_and_return_conditional_losses_137202004�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_layer_call_fn_137201955�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_layer_call_fn_137201968�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
E__inference_conv2d_layer_call_and_return_conditional_losses_137200886m9�6
/�,
*�'
inputs�����������
� "-�*
#� 
0���������@@@
� �
*__inference_conv2d_layer_call_fn_137200879`9�6
/�,
*�'
inputs�����������
� " ����������@@@�
F__inference_dense_1_layer_call_and_return_conditional_losses_137201920^wx0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_1_layer_call_fn_137201910Qwx0�-
&�#
!�
inputs����������
� "������������
D__inference_dense_layer_call_and_return_conditional_losses_137201901^op0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
)__inference_dense_layer_call_fn_137201891Qop0�-
&�#
!�
inputs����������
� "������������
F__inference_flatten_layer_call_and_return_conditional_losses_137201882Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
+__inference_flatten_layer_call_fn_137201876M0�-
&�#
!�
inputs����������
� "������������
W__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_137201871�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
<__inference_global_average_pooling2d_layer_call_fn_137201865wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_137200906�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_layer_call_fn_137200901�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_model_layer_call_and_return_conditional_losses_137199561����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p 

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
D__inference_model_layer_call_and_return_conditional_losses_137199741����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
D__inference_model_layer_call_and_return_conditional_losses_137200561����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p 

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
D__inference_model_layer_call_and_return_conditional_losses_137200872����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p

 
� "m�j
c�`
�
0/0����������
�
0/1����������
�
0/2����������
� �
)__inference_model_layer_call_fn_137197984����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p 

 
� "]�Z
�
0����������
�
1����������
�
2�����������
)__inference_model_layer_call_fn_137199381����������������������������������������������������������������������������opwxB�?
8�5
+�(
input_1�����������
p

 
� "]�Z
�
0����������
�
1����������
�
2�����������
)__inference_model_layer_call_fn_137200081����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p 

 
� "]�Z
�
0����������
�
1����������
�
2�����������
)__inference_model_layer_call_fn_137200250����������������������������������������������������������������������������opwxA�>
7�4
*�'
inputs�����������
p

 
� "]�Z
�
0����������
�
1����������
�
2�����������
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201166����������������G�D
-�*
(�%
inputs���������  @
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_1_layer_call_and_return_conditional_losses_137201224����������������G�D
-�*
(�%
inputs���������  @
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_1_layer_call_fn_137201073����������������G�D
-�*
(�%
inputs���������  @
�

trainingp "!������������
3__inference_residual_unit_1_layer_call_fn_137201108����������������G�D
-�*
(�%
inputs���������  @
�

trainingp"!������������
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201315�����������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_2_layer_call_and_return_conditional_losses_137201356�����������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_2_layer_call_fn_137201249�����������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_2_layer_call_fn_137201274�����������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201484����������������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_3_layer_call_and_return_conditional_losses_137201542����������������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_3_layer_call_fn_137201391����������������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_3_layer_call_fn_137201426����������������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201633�����������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_4_layer_call_and_return_conditional_losses_137201674�����������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_4_layer_call_fn_137201567�����������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_4_layer_call_fn_137201592�����������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201802����������������H�E
.�+
)�&
inputs����������
�

trainingp ".�+
$�!
0����������
� �
N__inference_residual_unit_5_layer_call_and_return_conditional_losses_137201860����������������H�E
.�+
)�&
inputs����������
�

trainingp".�+
$�!
0����������
� �
3__inference_residual_unit_5_layer_call_fn_137201709����������������H�E
.�+
)�&
inputs����������
�

trainingp "!������������
3__inference_residual_unit_5_layer_call_fn_137201744����������������H�E
.�+
)�&
inputs����������
�

trainingp"!������������
L__inference_residual_unit_layer_call_and_return_conditional_losses_137200997����������G�D
-�*
(�%
inputs���������  @
�

trainingp "-�*
#� 
0���������  @
� �
L__inference_residual_unit_layer_call_and_return_conditional_losses_137201038����������G�D
-�*
(�%
inputs���������  @
�

trainingp"-�*
#� 
0���������  @
� �
1__inference_residual_unit_layer_call_fn_137200931����������G�D
-�*
(�%
inputs���������  @
�

trainingp " ����������  @�
1__inference_residual_unit_layer_call_fn_137200956����������G�D
-�*
(�%
inputs���������  @
�

trainingp" ����������  @�
G__inference_sampling_layer_call_and_return_conditional_losses_137201942�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
,__inference_sampling_layer_call_fn_137201926y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
'__inference_signature_wrapper_137199912����������������������������������������������������������������������������opwxE�B
� 
;�8
6
input_1+�(
input_1�����������"���
)
dense �
dense����������
-
dense_1"�
dense_1����������
/
sampling#� 
sampling����������