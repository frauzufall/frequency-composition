// Pixelate a volume: block size `block.x` voxels along x and z,
// `block.y` voxels along y
#version 330
uniform sampler2D tex0;
uniform float size;
uniform vec2 block;
out vec4 fragColor;

void main( void )
{
	int n = int(size);
	int row = int(gl_FragCoord.y);
	vec3 voxel = vec3(int(gl_FragCoord.x), row % n, row / n);
	float bh = max(block.x, 1.);
	float bv = max(block.y, 1.);
	ivec3 q = ivec3(floor(voxel.x / bh) * bh, floor(voxel.y / bv) * bv, floor(voxel.z / bh) * bh);
	fragColor = texelFetch(tex0, ivec2(q.x, q.z * n + q.y), 0);
}
