
## 5. `.github/ISSUE_TEMPLATE/feature_request.md`

```markdown
---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement, needs-triage
assignees: ''
---

## Problem Statement
A clear and concise description of what problem this feature would solve. Ex. I'm always frustrated when [...]

## Proposed Solution
A clear and concise description of what you want to happen.

## Alternative Solutions
A clear and concise description of any alternative solutions or features you've considered.

## Use Cases
Describe specific use cases for this feature:

1. **Use Case 1:** [Description]
   - Actor: [Who uses it]
   - Trigger: [What initiates the use case]
   - Preconditions: [What must be true before]
   - Steps: [Detailed steps]
   - Postconditions: [Result after completion]
   - Success Criteria: [How to measure success]

2. **Use Case 2:** [Description]
   - Actor: [Who uses it]
   - Trigger: [What initiates the use case]
   - Preconditions: [What must be true before]
   - Steps: [Detailed steps]
   - Postconditions: [Result after completion]
   - Success Criteria: [How to measure success]

## Technical Specifications
**API Changes:**
```python
# Example API endpoint
@app.post("/api/v1/new_feature")
async def new_feature(request: NewFeatureRequest) -> NewFeatureResponse:
    """New feature endpoint."""
    pass

# Request schema
class NewFeatureRequest(BaseModel):
    param1: str
    param2: Optional[int] = None
    param3: List[float] = []

# Response schema
class NewFeatureResponse(BaseModel):
    result: Dict[str, Any]
    status: str
    processing_time: float