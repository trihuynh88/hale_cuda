#version 150 core
uniform mat4 projectMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
uniform vec4 colorSolid;
in vec4 positionVA;
in vec3 normalVA;
in vec2 tex2VA;
out vec3 norm_frag;
out vec4 color_frag;
out vec2 tex2VA_coor;
mat4 modIT = transpose(inverse(modelMat));
void main(void) {
  //some fake computations to avoid assertion in "unused uniforms"
  mat4 trans = projectMat * viewMat * modelMat;
  gl_Position = (trans*inverse(trans)) * positionVA;
  norm_frag = mat3(modIT) * normalVA;
  color_frag = colorSolid;
  tex2VA_coor = tex2VA;
}
