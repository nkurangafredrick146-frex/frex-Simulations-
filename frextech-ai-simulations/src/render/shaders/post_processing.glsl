// Post-Processing Shader Collection
// Includes: Bloom, Tonemapping, Color Grading, FXAA, Depth of Field, Motion Blur, SSR

#version 450 core

// Common uniforms
uniform sampler2D uMainTex;           // Main scene texture
uniform sampler2D uDepthTex;          // Depth texture
uniform sampler2D uNormalTex;         // Normal texture
uniform sampler2D uVelocityTex;       // Motion vectors
uniform sampler2D uBloomTex;          // Bloom texture
uniform vec2 uResolution;             // Screen resolution
uniform float uTime;                  // Time in seconds
uniform mat4 uInvProjection;          // Inverse projection matrix
uniform mat4 uInvView;                // Inverse view matrix
uniform mat4 uPrevViewProjection;     // Previous frame view-projection
uniform mat4 uViewProjection;         // Current view-projection
uniform vec3 uCameraPos;              // Camera position
uniform float uNearPlane = 0.1;       // Camera near plane
uniform float uFarPlane = 1000.0;     // Camera far plane
uniform float uExposure = 1.0;        // Exposure value
uniform float uGamma = 2.2;           // Gamma value
uniform int uEffectMode = 0;          // Post-processing effect mode

// Bloom parameters
uniform float uBloomThreshold = 1.0;
uniform float uBloomIntensity = 0.5;
uniform float uBloomRadius = 0.5;
uniform int uBloomSamples = 16;

// Tonemapping parameters
uniform int uTonemapMode = 2; // 0: None, 1: Reinhard, 2: ACES, 3: Filmic, 4: Uncharted2
uniform float uWhitePoint = 1.0;

// Color grading parameters
uniform vec3 uColorBalance = vec3(1.0); // RGB balance
uniform float uContrast = 1.0;
uniform float uSaturation = 1.0;
uniform float uBrightness = 0.0;
uniform vec3 uColorFilter = vec3(1.0);
uniform sampler2D uLUT; // Color lookup table

// FXAA parameters
uniform float uFXAAQuality = 1.0;
uniform float uFXAAEdgeThreshold = 0.125;
uniform float uFXAAEdgeThresholdMin = 0.03125;

// Depth of Field parameters
uniform float uDOFFocalDistance = 10.0;
uniform float uDOFAperture = 0.1;
uniform float uDOFFocalLength = 50.0;
uniform int uDOFSamples = 16;
uniform float uDOFMaxBlur = 8.0;

// Motion Blur parameters
uniform float uMotionBlurStrength = 0.5;
uniform int uMotionBlurSamples = 16;
uniform float uMotionBlurMaxVelocity = 32.0;

// Screen Space Reflections parameters
uniform float uSSRMaxDistance = 100.0;
uniform float uSSRThickness = 0.1;
uniform int uSSRSamples = 32;
uniform int uSSRBinarySearchSteps = 8;
uniform float uSSRRoughnessFade = 0.2;

// Vignette parameters
uniform float uVignetteIntensity = 0.5;
uniform float uVignetteRadius = 0.8;
uniform float uVignetteSoftness = 0.5;

// Chromatic Aberration parameters
uniform float uChromaticAberration = 0.0;
uniform vec2 uChromaticDirection = vec2(1.0, 0.0);

// Common functions
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * uNearPlane * uFarPlane) / (uFarPlane + uNearPlane - z * (uFarPlane - uNearPlane));
}

vec3 worldPosFromDepth(float depth, vec2 uv) {
    vec4 clipSpace = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 worldSpace = uInvView * uInvProjection * clipSpace;
    return worldSpace.xyz / worldSpace.w;
}

// Bloom effect
vec4 applyBloom(vec2 uv, vec4 color) {
    vec4 bloom = texture(uBloomTex, uv);
    
    // Gaussian blur approximation
    if (uBloomRadius > 0.0) {
        vec4 blur = vec4(0.0);
        float totalWeight = 0.0;
        
        for (int i = 0; i < uBloomSamples; i++) {
            float angle = float(i) * (2.0 * 3.14159265359 / float(uBloomSamples));
            vec2 offset = vec2(cos(angle), sin(angle)) * uBloomRadius / uResolution;
            
            float weight = 1.0 - abs(float(i) / float(uBloomSamples) - 0.5) * 2.0;
            weight = weight * weight; // Quadratic falloff
            
            blur += texture(uBloomTex, uv + offset) * weight;
            totalWeight += weight;
        }
        
        bloom = blur / totalWeight;
    }
    
    // Add bloom to color
    return color + bloom * uBloomIntensity;
}

// Tonemapping operators
vec3 tonemapReinhard(vec3 color) {
    return color / (color + vec3(1.0));
}

vec3 tonemapACES(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 tonemapFilmic(vec3 color) {
    vec3 x = max(vec3(0.0), color - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

vec3 tonemapUncharted2(vec3 color) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    
    vec3 x = color;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 applyTonemapping(vec3 color) {
    switch (uTonemapMode) {
        case 1: // Reinhard
            return tonemapReinhard(color * uExposure);
        case 2: // ACES
            return tonemapACES(color * uExposure);
        case 3: // Filmic
            return tonemapFilmic(color * uExposure);
        case 4: // Uncharted2
            return tonemapUncharted2(color * uExposure);
        default: // None
            return color * uExposure;
    }
}

// Color grading
vec3 applyColorGrading(vec3 color) {
    // Color balance
    color *= uColorBalance;
    
    // Brightness
    color += uBrightness;
    
    // Contrast
    color = (color - 0.5) * uContrast + 0.5;
    
    // Saturation
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, uSaturation);
    
    // Color filter
    color *= uColorFilter;
    
    // LUT (simplified 3D LUT sampling)
    if (textureSize(uLUT, 0).x > 1) {
        // Assuming 32x32x32 LUT stored as 32x1024 2D texture
        const float lutSize = 32.0;
        const float cellSize = 1.0 / lutSize;
        
        vec3 lutCoord = color * (lutSize - 1.0) / lutSize;
        float blueFraction = fract(lutCoord.b * (lutSize - 1.0));
        
        // Get two slices
        vec2 slice0 = vec2(
            floor(lutCoord.b * (lutSize - 1.0)) / lutSize,
            0.0
        );
        vec2 slice1 = slice0 + vec2(cellSize, 0.0);
        
        // Sample from both slices and interpolate
        vec2 uv0 = vec2(
            lutCoord.r * cellSize + slice0.x + lutCoord.g * lutSize * cellSize,
            slice0.y
        );
        vec2 uv1 = vec2(
            lutCoord.r * cellSize + slice1.x + lutCoord.g * lutSize * cellSize,
            slice1.y
        );
        
        vec3 color0 = texture(uLUT, uv0).rgb;
        vec3 color1 = texture(uLUT, uv1).rgb;
        
        color = mix(color0, color1, blueFraction);
    }
    
    return color;
}

// FXAA (Fast Approximate Anti-Aliasing)
vec4 applyFXAA(vec2 uv) {
    // Based on NVIDIA FXAA 3.11
    vec2 invRes = 1.0 / uResolution;
    
    // Luma coefficients
    const vec3 luma = vec3(0.299, 0.587, 0.114);
    
    float lumaCenter = dot(texture(uMainTex, uv).rgb, luma);
    float lumaDown = dot(textureOffset(uMainTex, uv, ivec2(0, -1)).rgb, luma);
    float lumaUp = dot(textureOffset(uMainTex, uv, ivec2(0, 1)).rgb, luma);
    float lumaLeft = dot(textureOffset(uMainTex, uv, ivec2(-1, 0)).rgb, luma);
    float lumaRight = dot(textureOffset(uMainTex, uv, ivec2(1, 0)).rgb, luma);
    
    float lumaMin = min(lumaCenter, min(min(lumaDown, lumaUp), min(lumaLeft, lumaRight)));
    float lumaMax = max(lumaCenter, max(max(lumaDown, lumaUp), max(lumaLeft, lumaRight)));
    
    float lumaRange = lumaMax - lumaMin;
    
    // Early exit if contrast is low
    if (lumaRange < max(uFXAAEdgeThresholdMin, lumaMax * uFXAAEdgeThreshold)) {
        return texture(uMainTex, uv);
    }
    
    // Calculate blending weights
    float lumaDownLeft = dot(textureOffset(uMainTex, uv, ivec2(-1, -1)).rgb, luma);
    float lumaUpRight = dot(textureOffset(uMainTex, uv, ivec2(1, 1)).rgb, luma);
    float lumaUpLeft = dot(textureOffset(uMainTex, uv, ivec2(-1, 1)).rgb, luma);
    float lumaDownRight = dot(textureOffset(uMainTex, uv, ivec2(1, -1)).rgb, luma);
    
    float lumaDownUp = lumaDown + lumaUp;
    float lumaLeftRight = lumaLeft + lumaRight;
    float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
    float lumaDownCorners = lumaDownLeft + lumaDownRight;
    float lumaRightCorners = lumaDownRight + lumaUpRight;
    float lumaUpCorners = lumaUpRight + lumaUpLeft;
    
    float edgeHorizontal = abs(lumaDownCorners - 2.0 * lumaDown) + 
                          abs(lumaUpCorners - 2.0 * lumaUp) * 2.0 + 
                          abs(lumaLeftCorners - 2.0 * lumaLeft);
    float edgeVertical = abs(lumaLeftCorners - 2.0 * lumaLeft) + 
                        abs(lumaRightCorners - 2.0 * lumaRight) * 2.0 + 
                        abs(lumaDownCorners - 2.0 * lumaDown);
    
    bool isHorizontal = edgeHorizontal >= edgeVertical;
    
    float luma1 = isHorizontal ? lumaDown : lumaLeft;
    float luma2 = isHorizontal ? lumaUp : lumaRight;
    float gradient1 = luma1 - lumaCenter;
    float gradient2 = luma2 - lumaCenter;
    
    bool is1Steepest = abs(gradient1) >= abs(gradient2);
    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));
    
    vec2 stepSize = isHorizontal ? vec2(invRes.x, 0.0) : vec2(0.0, invRes.y);
    
    if (is1Steepest) {
        stepSize = -stepSize;
    }
    
    vec2 uv1 = uv + stepSize;
    vec2 uv2 = uv - stepSize;
    
    // Sub-pixel blending
    float lumaEnd1 = dot(texture(uMainTex, uv1).rgb, luma);
    float lumaEnd2 = dot(texture(uMainTex, uv2).rgb, luma);
    lumaEnd1 -= lumaCenter * 0.5;
    lumaEnd2 -= lumaCenter * 0.5;
    
    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;
    bool reachedBoth = reached1 && reached2;
    
    if (!reached1) uv1 -= stepSize;
    if (!reached2) uv2 -= stepSize;
    
    // Final blending
    if (reachedBoth) {
        vec3 color1 = texture(uMainTex, uv1).rgb;
        vec3 color2 = texture(uMainTex, uv2).rgb;
        vec3 color = (color1 + color2) * 0.5;
        return vec4(color, 1.0);
    } else {
        vec3 color = texture(uMainTex, uv).rgb;
        return vec4(color, 1.0);
    }
}

// Depth of Field (Circle of Confusion based)
vec4 applyDepthOfField(vec2 uv, vec4 color) {
    float depth = texture(uDepthTex, uv).r;
    float linearDepth = linearizeDepth(depth);
    
    // Calculate Circle of Confusion (CoC) radius
    float coc = abs(linearDepth - uDOFFocalDistance) / uDOFFocalDistance;
    coc = coc * uDOFAperture * (uDOFFocalLength / linearDepth);
    coc = clamp(coc, 0.0, uDOFMaxBlur) / uResolution.x;
    
    if (coc < 0.001) {
        return color;
    }
    
    // Poisson disc sampling for bokeh
    const vec2 poissonDisk[16] = vec2[](
        vec2(-0.942, -0.399), vec2(-0.696,  0.457),
        vec2(-0.203,  0.962), vec2( 0.962, -0.194),
        vec2( 0.473, -0.480), vec2( 0.519,  0.767),
        vec2( 0.185, -0.893), vec2( 0.507,  0.064),
        vec2( 0.896,  0.412), vec2(-0.322, -0.933),
        vec2(-0.792, -0.598), vec2(-0.559,  0.359),
        vec2(-0.985,  0.127), vec2( 0.140,  0.980),
        vec2(-0.244,  0.724), vec2( 0.671,  0.338)
    );
    
    vec4 blur = vec4(0.0);
    float totalWeight = 0.0;
    
    for (int i = 0; i < uDOFSamples; i++) {
        vec2 offset = poissonDisk[i] * coc;
        vec2 sampleUV = uv + offset;
        
        // Check if sample is in bounds
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            continue;
        }
        
        float sampleDepth = texture(uDepthTex, sampleUV).r;
        float sampleLinearDepth = linearizeDepth(sampleDepth);
        
        // Weight based on depth difference
        float weight = exp(-abs(sampleLinearDepth - linearDepth) * 10.0);
        
        blur += texture(uMainTex, sampleUV) * weight;
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0) {
        blur /= totalWeight;
        
        // Mix based on CoC size
        float mixFactor = smoothstep(0.0, uDOFMaxBlur / uResolution.x, coc);
        return mix(color, blur, mixFactor);
    }
    
    return color;
}

// Motion Blur
vec4 applyMotionBlur(vec2 uv, vec4 color) {
    vec2 velocity = texture(uVelocityTex, uv).rg;
    float speed = length(velocity);
    
    if (speed < 0.001) {
        return color;
    }
    
    // Clamp velocity
    velocity = normalize(velocity) * min(speed, uMotionBlurMaxVelocity) * uMotionBlurStrength;
    
    vec4 blur = color;
    int samples = int(ceil(speed * float(uMotionBlurSamples)));
    samples = clamp(samples, 1, uMotionBlurSamples);
    
    for (int i = 1; i < samples; i++) {
        float t = float(i) / float(samples - 1) - 0.5;
        vec2 offset = velocity * t;
        vec2 sampleUV = uv + offset;
        
        if (sampleUV.x >= 0.0 && sampleUV.x <= 1.0 && sampleUV.y >= 0.0 && sampleUV.y <= 1.0) {
            blur += texture(uMainTex, sampleUV);
        }
    }
    
    blur /= float(samples);
    return blur;
}

// Screen Space Reflections
vec3 calculateSSR(vec2 uv, vec3 position, vec3 normal, vec3 viewDir, float roughness) {
    if (roughness > uSSRRoughnessFade) {
        return vec3(0.0);
    }
    
    vec3 reflectionDir = reflect(viewDir, normal);
    
    // Ray marching in screen space
    vec3 rayPos = position;
    vec3 rayStep = reflectionDir * (uSSRMaxDistance / float(uSSRSamples));
    
    float minHitDistance = uSSRMaxDistance;
    vec2 hitUV = uv;
    
    for (int i = 0; i < uSSRSamples; i++) {
        rayPos += rayStep;
        
        // Project ray position to screen space
        vec4 clipPos = uViewProjection * vec4(rayPos, 1.0);
        vec3 ndc = clipPos.xyz / clipPos.w;
        vec2 screenPos = ndc.xy * 0.5 + 0.5;
        
        // Check if out of screen
        if (screenPos.x < 0.0 || screenPos.x > 1.0 || screenPos.y < 0.0 || screenPos.y > 1.0) {
            break;
        }
        
        // Get depth at sample point
        float sampleDepth = texture(uDepthTex, screenPos).r;
        float linearSampleDepth = linearizeDepth(sampleDepth);
        
        // Get world position at sample point
        vec3 samplePos = worldPosFromDepth(sampleDepth, screenPos);
        
        // Check for intersection
        float rayDistance = length(rayPos - position);
        float depthDifference = rayPos.z - samplePos.z;
        
        if (depthDifference > 0.0 && depthDifference < uSSRThickness && rayDistance < minHitDistance) {
            minHitDistance = rayDistance;
            hitUV = screenPos;
            
            // Binary search for better accuracy
            vec3 binaryStep = rayStep * 0.5;
            vec3 binaryPos = rayPos - rayStep;
            
            for (int j = 0; j < uSSRBinarySearchSteps; j++) {
                binaryStep *= 0.5;
                
                vec4 binaryClipPos = uViewProjection * vec4(binaryPos, 1.0);
                vec3 binaryNDC = binaryClipPos.xyz / binaryClipPos.w;
                vec2 binaryScreenPos = binaryNDC.xy * 0.5 + 0.5;
                
                float binaryDepth = texture(uDepthTex, binaryScreenPos).r;
                vec3 binarySamplePos = worldPosFromDepth(binaryDepth, binaryScreenPos);
                
                float binaryDepthDiff = binaryPos.z - binarySamplePos.z;
                
                if (binaryDepthDiff > 0.0) {
                    binaryPos -= binaryStep;
                } else {
                    binaryPos += binaryStep;
                }
            }
            
            // Final hit position
            vec4 finalClipPos = uViewProjection * vec4(binaryPos, 1.0);
            vec3 finalNDC = finalClipPos.xyz / finalClipPos.w;
            hitUV = finalNDC.xy * 0.5 + 0.5;
            break;
        }
    }
    
    if (minHitDistance < uSSRMaxDistance) {
        // Sample reflection color
        vec3 reflectionColor = texture(uMainTex, hitUV).rgb;
        
        // Fade based on distance and roughness
        float fade = 1.0 - smoothstep(0.0, uSSRMaxDistance, minHitDistance);
        fade *= 1.0 - smoothstep(0.0, uSSRRoughnessFade, roughness);
        
        return reflectionColor * fade;
    }
    
    return vec3(0.0);
}

// Vignette effect
vec3 applyVignette(vec2 uv, vec3 color) {
    if (uVignetteIntensity <= 0.0) {
        return color;
    }
    
    vec2 center = uv - 0.5;
    float dist = length(center) * 2.0;
    float vignette = 1.0 - smoothstep(uVignetteRadius, uVignetteRadius + uVignetteSoftness, dist);
    vignette = mix(1.0, vignette, uVignetteIntensity);
    
    return color * vignette;
}

// Chromatic Aberration
vec3 applyChromaticAberration(vec2 uv, vec3 color) {
    if (uChromaticAberration <= 0.0) {
        return color;
    }
    
    vec2 direction = normalize(uChromaticDirection) * uChromaticAberration / uResolution;
    
    float r = texture(uMainTex, uv + direction).r;
    float g = texture(uMainTex, uv).g;
    float b = texture(uMainTex, uv - direction).b;
    
    return vec3(r, g, b);
}

// Main fragment shader
in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    vec2 uv = vTexCoord;
    vec4 color = texture(uMainTex, uv);
    
    // Early depth test for skybox
    float depth = texture(uDepthTex, uv).r;
    if (depth >= 0.999) {
        fragColor = color;
        return;
    }
    
    // Apply selected effect
    switch (uEffectMode) {
        case 0: // Full pipeline
            // Tonemapping
            color.rgb = applyTonemapping(color.rgb);
            
            // Color grading
            color.rgb = applyColorGrading(color.rgb);
            
            // Bloom
            color = applyBloom(uv, color);
            
            // Depth of Field
            color = applyDepthOfField(uv, color);
            
            // Motion Blur
            color = applyMotionBlur(uv, color);
            
            // Vignette
            color.rgb = applyVignette(uv, color.rgb);
            
            // Chromatic Aberration
            color.rgb = applyChromaticAberration(uv, color.rgb);
            
            // Gamma correction
            color.rgb = pow(color.rgb, vec3(1.0 / uGamma));
            break;
            
        case 1: // FXAA only
            color = applyFXAA(uv);
            break;
            
        case 2: // Bloom only
            color = applyBloom(uv, color);
            break;
            
        case 3: // Depth of Field only
            color = applyDepthOfField(uv, color);
            break;
            
        case 4: // Motion Blur only
            color = applyMotionBlur(uv, color);
            break;
            
        case 5: // SSR only
            vec3 position = worldPosFromDepth(depth, uv);
            vec3 normal = texture(uNormalTex, uv).rgb * 2.0 - 1.0;
            vec3 viewDir = normalize(uCameraPos - position);
            
            // Get roughness from metallic-roughness texture or uniform
            float roughness = 0.5; // Default
            
            vec3 ssr = calculateSSR(uv, position, normal, viewDir, roughness);
            color.rgb += ssr;
            break;
            
        case 6: // Color grading only
            color.rgb = applyColorGrading(color.rgb);
            color.rgb = applyTonemapping(color.rgb);
            color.rgb = pow(color.rgb, vec3(1.0 / uGamma));
            break;
            
        default:
            break;
    }
    
    // Ensure valid alpha
    color.a = clamp(color.a, 0.0, 1.0);
    
    fragColor = color;
    
    // Debug outputs
    #ifdef DEBUG_DEPTH
        float linearDepth = linearizeDepth(depth);
        fragColor = vec4(vec3(linearDepth / uFarPlane), 1.0);
    #endif
    
    #ifdef DEBUG_NORMALS
        vec3 normal = texture(uNormalTex, uv).rgb;
        fragColor = vec4(normal * 0.5 + 0.5, 1.0);
    #endif
    
    #ifdef DEBUG_VELOCITY
        vec2 velocity = texture(uVelocityTex, uv).rg;
        fragColor = vec4(abs(velocity), 0.0, 1.0);
    #endif
}
