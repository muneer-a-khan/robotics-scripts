
ANNOTATION GUIDELINES FOR BETTER MODEL TRAINING

KEY PRINCIPLE: Label COMPLETE COMPONENTS, not individual parts!

CORRECT Annotation Strategy:
================================

BATTERY_HOLDER (class 0):
   - Draw ONE box around the entire battery compartment
   - Include all battery-related parts as one component

SWITCH (class 1): 
   - Draw ONE box around the complete switch assembly
   - Don't separate switch parts - treat as single unit

LED (class 2):
   - Draw box around the complete LED component
   - Include the LED housing and immediate connections

CONNECTION (class 3):
   - Draw ONE large box covering connected wire segments  
   - Think: "Where does current flow as one path?"
   - Group connected points into single component

COMMON MISTAKES TO AVOID:
==========================

- Don't label individual connection points separately
- Don't split switches into multiple parts  
- Don't create tiny boxes for wire segments
- Don't over-segment what should be one component

RESULT: Model learns to recognize complete functional units!
