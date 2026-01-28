// Gaussian Splatting Shader for 3D Gaussian Splat Rendering
// Based on 3D Gaussian Splatting for Real-Time Radiance Field Rendering

#version 450 core

// Input from vertex shader
in VertexData {
    vec3 worldPos;
    vec3 viewPos;
    vec2 screenPos;
    vec3 color;
    float opacity;
    vec2 cov2dA;
    vec2 cov2dB;
    float scale;
    float depth;
} vData;

// Output
out vec4 FragColor;

// Uniforms
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uInvProjectionMatrix;
uniform vec3 uCameraPos;
uniform vec2 uViewportSize;
uniform float uFocalLength;
uniform float uFocalLengthX;
uniform float uFocalLengthY;
uniform vec2 uPrincipalPoint;
uniform float uScaleModifier = 1.0;
uniform int uShadingMode = 0; // 0: Lambertian, 1: Spherical Harmonics, 2: Textured
uniform int uBlendMode = 1; // 0: Alpha blending, 1: Weighted blending
uniform float uMinAlpha = 0.01;

// SH coefficients (if using spherical harmonics)
#define SH_DEGREE 3
#define SH_COEFF_COUNT ((SH_DEGREE + 1) * (SH_DEGREE + 1))
uniform vec3 uSHCoeffs[SH_COEFF_COUNT];

// Lighting
uniform vec3 uLightPos = vec3(0.0, 10.0, 0.0);
uniform vec3 uLightColor = vec3(1.0, 1.0, 1.0);
uniform float uAmbientStrength = 0.3;
uniform float uDiffuseStrength = 0.7;
uniform float uSpecularStrength = 0.0;
uniform float uShininess = 32.0;

// Textures
uniform sampler2D uColorTexture;
uniform sampler2D uOpacityTexture;
uniform sampler2D uScaleTexture;
uniform sampler2D uRotationTexture;

// Gaussian functions
float gaussian2D(vec2 x, vec2 mu, mat2 sigma) {
    float det = determinant(sigma);
    if (det <= 0.0) return 0.0;
    
    mat2 invSigma = inverse(sigma);
    vec2 diff = x - mu;
    float exponent = -0.5 * dot(diff, invSigma * diff);
    
    return exp(exponent) / (2.0 * 3.14159265359 * sqrt(det));
}

float computeGaussianWeight(vec2 pixelCoord, vec2 center, mat2 covariance) {
    // Convert covariance from world space to screen space
    vec2 diff = pixelCoord - center;
    float weight = exp(-0.5 * dot(diff, covariance * diff));
    
    // Apply circular falloff
    float dist = length(diff);
    float radius = sqrt(1.0 / min(covariance[0][0], covariance[1][1]));
    
    if (dist > radius) {
        weight = 0.0;
    }
    
    return weight;
}

mat3 computeCovariance3D(vec3 scale, vec4 rotation) {
    // Build scaling matrix
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    
    // Convert quaternion to rotation matrix
    float x = rotation.x, y = rotation.y, z = rotation.z, w = rotation.w;
    mat3 R = mat3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
        2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
        2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
    );
    
    // Covariance = R * S * S^T * R^T
    mat3 M = R * S;
    return M * transpose(M);
}

mat2 projectCovariance3DTo2D(mat3 cov3d, vec3 viewPos) {
    // Project 3D covariance to 2D screen space
    // J is Jacobian of the projective transformation
    float focalX = uFocalLengthX;
    float focalY = uFocalLengthY;
    
    // Limit near plane to avoid singularities
    float z = max(viewPos.z, 0.0001);
    
    // Jacobian matrix
    mat3 J = mat3(
        focalX / z, 0.0, -focalX * viewPos.x / (z * z),
        0.0, focalY / z, -focalY * viewPos.y / (z * z),
        0.0, 0.0, 0.0
    );
    
    // Projected covariance: J * cov3d * J^T
    mat3 W = transpose(J) * cov3d * J;
    
    // Extract 2x2 upper-left submatrix
    return mat2(W[0][0], W[0][1], W[1][0], W[1][1]);
}

vec3 evaluateSH(vec3 dir, vec3 shCoeffs[SH_COEFF_COUNT]) {
    // Evaluate spherical harmonics up to degree 3
    vec3 color = vec3(0.0);
    
    // L0 (degree 0)
    color += 0.2820947918 * shCoeffs[0];
    
    // L1 (degree 1)
    color += 0.4886025119 * dir.y * shCoeffs[1];
    color += 0.4886025119 * dir.z * shCoeffs[2];
    color += 0.4886025119 * dir.x * shCoeffs[3];
    
    // L2 (degree 2)
    color += 1.0925484306 * dir.x * dir.y * shCoeffs[4];
    color += 1.0925484306 * dir.y * dir.z * shCoeffs[5];
    color += 0.3153915652 * (3.0 * dir.z * dir.z - 1.0) * shCoeffs[6];
    color += 1.0925484306 * dir.x * dir.z * shCoeffs[7];
    color += 0.5462742153 * (dir.x * dir.x - dir.y * dir.y) * shCoeffs[8];
    
    // L3 (degree 3)
    color += 0.590043586 * dir.y * (3.0 * dir.x * dir.x - dir.y * dir.y) * shCoeffs[9];
    color += 2.890611421 * dir.x * dir.y * dir.z * shCoeffs[10];
    color += 0.4570458 * dir.y * (5.0 * dir.z * dir.z - 1.0) * shCoeffs[11];
    color += 0.37317633 * (5.0 * dir.z * dir.z - 3.0) * dir.z * shCoeffs[12];
    color += 0.4570458 * dir.x * (5.0 * dir.z * dir.z - 1.0) * shCoeffs[13];
    color += 1.44530571 * (dir.x * dir.x - dir.y * dir.y) * dir.z * shCoeffs[14];
    color += 0.590043586 * dir.x * (dir.x * dir.x - 3.0 * dir.y * dir.y) * shCoeffs[15];
    
    return max(color, vec3(0.0));
}

vec3 computeLighting(vec3 position, vec3 normal, vec3 color, vec3 viewDir) {
    // Ambient
    vec3 ambient = uAmbientStrength * color;
    
    // Diffuse
    vec3 lightDir = normalize(uLightPos - position);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = uDiffuseStrength * diff * color * uLightColor;
    
    // Specular (Blinn-Phong)
    vec3 specular = vec3(0.0);
    if (uSpecularStrength > 0.0) {
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), uShininess);
        specular = uSpecularStrength * spec * uLightColor;
    }
    
    return ambient + diffuse + specular;
}

void main() {
    // Early discard if opacity is too low
    if (vData.opacity < uMinAlpha) {
        discard;
    }
    
    // Get pixel coordinates in screen space
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 pixelCenter = vData.screenPos * uViewportSize;
    
    // Compute 2D covariance matrix from passed coefficients
    mat2 cov2d = mat2(
        vData.cov2dA.x, vData.cov2dA.y,
        vData.cov2dB.x, vData.cov2dB.y
    );
    
    // Compute Gaussian weight
    float weight = computeGaussianWeight(fragCoord, pixelCenter, cov2d);
    
    // Apply opacity
    float alpha = vData.opacity * weight;
    
    // Early discard if effectively transparent
    if (alpha < 0.001) {
        discard;
    }
    
    // Base color
    vec3 baseColor = vData.color;
    
    // Apply shading based on mode
    vec3 finalColor = baseColor;
    
    if (uShadingMode == 1) {
        // Spherical Harmonics shading
        vec3 viewDir = normalize(uCameraPos - vData.worldPos);
        vec3 shColor = evaluateSH(viewDir, uSHCoeffs);
        finalColor = baseColor * shColor;
    } else if (uShadingMode == 2) {
        // Textured shading (sample from texture)
        vec2 uv = gl_FragCoord.xy / uViewportSize;
        finalColor = texture(uColorTexture, uv).rgb;
    }
    
    // Apply lighting if in appropriate mode
    if (uShadingMode == 0 || uShadingMode == 2) {
        // Compute normal (approximated from view direction for splats)
        vec3 normal = normalize(vData.worldPos - uCameraPos);
        vec3 viewDir = normalize(uCameraPos - vData.worldPos);
        finalColor = computeLighting(vData.worldPos, normal, finalColor, viewDir);
    }
    
    // Depth-based fade for better blending
    float depthFade = 1.0 - smoothstep(0.0, 10.0, vData.depth);
    alpha *= depthFade;
    
    // Apply blending
    if (uBlendMode == 0) {
        // Alpha blending: src * alpha + dst * (1 - alpha)
        FragColor = vec4(finalColor, alpha);
    } else {
        // Weighted blending (for order-independent transparency)
        float weight = alpha * max(0.0, min(1.0, 1.0 - alpha));
        FragColor = vec4(finalColor * weight, alpha);
    }
    
    // Ensure valid alpha
    FragColor.a = clamp(FragColor.a, 0.0, 1.0);
    
    // Debug visualization modes
    #ifdef DEBUG_COVARIANCE
        // Visualize covariance as ellipses
        vec2 diff = fragCoord - pixelCenter;
        float covValue = dot(diff, cov2d * diff);
        FragColor = mix(vec4(1,0,0,1), vec4(0,1,0,1), exp(-covValue));
    #endif
    
    #ifdef DEBUG_DEPTH
        // Visualize depth
        float normalizedDepth = clamp(vData.depth / 100.0, 0.0, 1.0);
        FragColor = vec4(vec3(normalizedDepth), 1.0);
    #endif
    
    #ifdef DEBUG_OPACITY
        // Visualize opacity
        FragColor = vec4(vec3(vData.opacity), 1.0);
    #endif
}

// Optional: Geometry shader for splat expansion
#ifdef WITH_GEOMETRY_SHADER
#version 450 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in SplatData {
    vec3 position;
    vec3 color;
    vec3 scale;
    vec4 rotation;
    float opacity;
    vec3 shCoeffs[SH_COEFF_COUNT];
} gsIn[];

out VertexData {
    vec3 worldPos;
    vec3 viewPos;
    vec2 screenPos;
    vec3 color;
    float opacity;
    vec2 cov2dA;
    vec2 cov2dB;
    float scale;
    float depth;
} gsOut;

void emitVertex(vec2 offset, vec2 texCoord) {
    // Transform offset by scale and rotation
    vec3 worldOffset = vec3(offset.x * gsIn[0].scale.x, 
                           offset.y * gsIn[0].scale.y, 0.0);
    
    // Apply rotation (simplified 2D rotation for billboard)
    vec4 rot = gsIn[0].rotation;
    mat3 R = mat3(
        1.0 - 2.0*(rot.y*rot.y + rot.z*rot.z), 2.0*(rot.x*rot.y - rot.w*rot.z), 2.0*(rot.x*rot.z + rot.w*rot.y),
        2.0*(rot.x*rot.y + rot.w*rot.z), 1.0 - 2.0*(rot.x*rot.x + rot.z*rot.z), 2.0*(rot.y*rot.z - rot.w*rot.x),
        2.0*(rot.x*rot.z - rot.w*rot.y), 2.0*(rot.y*rot.z + rot.w*rot.x), 1.0 - 2.0*(rot.x*rot.x + rot.y*rot.y)
    );
    
    worldOffset = R * worldOffset;
    vec3 worldPos = gsIn[0].position + worldOffset;
    
    // Transform to view and clip space
    vec4 viewPos = uViewMatrix * vec4(worldPos, 1.0);
    vec4 clipPos = uProjectionMatrix * viewPos;
    
    // Pass data to fragment shader
    gsOut.worldPos = worldPos;
    gsOut.viewPos = viewPos.xyz;
    gsOut.screenPos = clipPos.xy / clipPos.w;
    gsOut.color = gsIn[0].color;
    gsOut.opacity = gsIn[0].opacity;
    gsOut.scale = length(gsIn[0].scale);
    gsOut.depth = -viewPos.z;
    
    // Compute and pass covariance (simplified)
    mat3 cov3d = computeCovariance3D(gsIn[0].scale, gsIn[0].rotation);
    mat2 cov2d = projectCovariance3DTo2D(cov3d, viewPos.xyz);
    
    gsOut.cov2dA = cov2d[0];
    gsOut.cov2dB = cov2d[1];
    
    gl_Position = clipPos;
    EmitVertex();
}

void main() {
    // Emit quad vertices
    emitVertex(vec2(-1, -1), vec2(0, 0));
    emitVertex(vec2(1, -1), vec2(1, 0));
    emitVertex(vec2(-1, 1), vec2(0, 1));
    emitVertex(vec2(1, 1), vec2(1, 1));
    
    EndPrimitive();
}
#endif
