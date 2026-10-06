// Colour oscillation for a volume that is stored flattened in a
// 2D texture: x = voxel x, y = z * size + voxel y.  `axis` (0=x, 1=y, 2=z) selects
// the direction the wave runs along.
#version 330
uniform float amplitude;
uniform float wavelength;
uniform float phase;
uniform float size;
uniform vec4 color1;
uniform vec4 color2;
uniform int axis;
out vec4 fragColor;

const float pi = 3.14159265;

void main( void ){
	int n = int(size);
	int row = int(gl_FragCoord.y);
	ivec3 voxel = ivec3(int(gl_FragCoord.x), row % n, row / n);
	float pos = float(voxel[axis]) + 0.5;

	float val = cos((pos / size * 2. * pi) / wavelength + phase) * 0.5 * amplitude + 0.5;

	fragColor = color1 + (color2 - color1) * val;
}
