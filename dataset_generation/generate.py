import json
import time
import os
import random
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=dotenv_path) # <--- ADD THIS

def generate_cat_basic_math_prompt():
    func_type = random.choice(["polynomial", "exponential", "trigonometric", "logarithmic"])
    request = ""
    if func_type == "polynomial":
        degree = random.randint(2, 4)
        coeffs = [round(random.uniform(-5, 5), 1) for _ in range(degree + 1)]
        if coeffs[0] == 0: coeffs[0] = 1.0 # Ensure highest degree coeff is not zero
        request = f"Plot a {degree}-degree polynomial with coefficients approximately {coeffs}. Use a suitable time vector to show its shape, perhaps from t=-5 to t=5."
    elif func_type == "exponential":
        base_val = round(random.uniform(0.5, 2.0), 1)
        rate = round(random.uniform(-1.5, 1.5), 2) # Can be decay or growth
        if rate == 0: rate = 0.1
        request = f"Plot an exponential function like y = {base_val}*exp({rate}*t). Choose a time vector (e.g., 0 to 5, or -3 to 3) that clearly shows its behavior."
    elif func_type == "trigonometric":
        trig_func = random.choice(["sin", "cos", "tan (one period)"])
        amplitude = round(random.uniform(0.5, 3.0), 1)
        frequency_hz = round(random.uniform(0.1, 2.0), 1)
        phase_shift_rad = round(random.uniform(0, 1.57), 2) # 0 to pi/2
        if "tan" in trig_func:
             request = f"Plot one period of a {trig_func} function. Use ylim to manage asymptotes if plotting tan."
        else:
            request = f"Plot a {trig_func} wave with amplitude around {amplitude}, frequency {frequency_hz} Hz, and a phase shift of {phase_shift_rad} radians. Show 2-3 cycles."
    elif func_type == "logarithmic":
        log_base = random.choice(["natural log (log)", "base-10 log (log10)"])
        request = f"Plot a {log_base} function. Ensure the time vector is appropriate for the domain of the logarithm (e.g., t > 0)."
    return {"category": "Basic Mathematical Functions", "request_template": request}

def generate_cat_transfer_function_prompt():
    sys_order = random.choice(["first-order", "second-order"])
    resp_type = random.choice(["step response", "impulse response"])
    request = ""
    if sys_order == "first-order":
        time_constant = round(random.uniform(0.2, 5.0), 1)
        gain = round(random.uniform(0.5, 5.0), 1)
        request = f"Generate a {sys_order} transfer function (e.g., G(s) = K/(Ts+1)) with gain K approx {gain} and time constant T approx {time_constant}s. Plot its {resp_type}. Time vector should show settling."
    elif sys_order == "second-order":
        wn = round(random.uniform(0.5, 10.0), 1)
        zeta = round(random.uniform(0.1, 1.5), 2) # under, critical, over
        gain = round(random.uniform(0.5, 3.0), 1)
        damping_desc = "underdamped" if zeta < 1 else ("critically damped" if zeta == 1 else "overdamped")
        request = f"Generate a {sys_order} {damping_desc} transfer function (e.g., G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)) with DC gain K approx {gain}, natural frequency wn approx {wn} rad/s, and damping ratio zeta approx {zeta}. Plot its {resp_type}. Choose plot duration to show key dynamics."
    return {"category": "Transfer Function Responses", "request_template": request}

def generate_cat_state_space_prompt():
    sys_order = random.choice(["first-order", "second-order"])
    resp_type = random.choice(["step response", "impulse response"])
    request = ""
    if sys_order == "first-order":
        pole_location = round(random.uniform(-5.0, -0.2), 1)
        gain_effect = round(random.uniform(0.5, 2.0),1) # Affects C or B matrix
        request = f"Define a stable {sys_order} state-space model (A,B,C,D matrices) with a dominant pole near s={pole_location} and an effective DC gain around {gain_effect}. Plot its {resp_type}. Plot for 5-10 seconds."
    elif sys_order == "second-order":
        wn = round(random.uniform(1.0, 6.0), 1)
        zeta = round(random.uniform(0.1, 1.2), 2)
        gain_effect = round(random.uniform(0.5, 1.5),1)
        damping_desc = "underdamped" if zeta < 1 else ("critically damped" if zeta == 1 else "overdamped")
        request = f"Define a {sys_order} {damping_desc} state-space model (A,B,C,D) with natural frequency approx {wn} rad/s, damping approx {zeta}, and DC gain effect of {gain_effect}. Plot its {resp_type}. Plot for 10-20 seconds."
    return {"category": "State-Space Model Responses", "request_template": request}

def generate_cat_lsim_prompt():
    input_type = random.choice([
        f"a unit step input u=ones(size(t))",
        f"a ramp input u=t* {round(random.uniform(0.2, 1.0),1)}",
        f"a sinusoidal input u={round(random.uniform(0.5,2.0),1)}*sin({round(random.uniform(0.5,2.0),1)}*t)",
        f"a pulse input (e.g., 1 for 0 to {random.randint(1,3)}s, then 0 for the rest)",
        f"zero input with non-zero initial conditions (e.g., x0 = [{round(random.uniform(-1,1),1)}; {round(random.uniform(-1,1),1)}]) for a 2nd order system"
    ])
    sys_type = random.choice(["first-order stable transfer function", "second-order underdamped state-space model", "second-order overdamped transfer function"])
    plot_duration = random.randint(10, 25)
    request = f"Define a {sys_type}. Simulate and plot its response using lsim to {input_type} for {plot_duration} seconds. Define time vector t appropriately."
    if "initial conditions" in input_type:
         request += " Ensure the lsim call correctly uses the specified initial conditions."
    return {"category": "LSIM with Various Inputs", "request_template": request}

def generate_cat_frequency_domain_prompt():
    plot_type = random.choice(["Bode plot (magnitude and phase)", "Nyquist diagram", "Nichols chart"])
    num_poles = random.randint(1, 3)
    num_zeros = random.randint(0, num_poles) # Zeros can be up to num_poles
    
    stability_req = "Try to make it stable or marginally stable." if "Nyquist" in plot_type or "Nichols" in plot_type else ""
    
    request = f"Generate a transfer function (tf object) with {num_poles} pole(s) and {num_zeros} zero(s). {stability_req} Then, display its {plot_type}. For Bode, use a frequency range that shows key features (e.g., using logspace). For Nyquist, clearly show encirclements if any or mark -1+j0 point."
    return {"category": "Frequency Domain Plots", "request_template": request}

# List of your category generator functions
category_generators = [
    generate_cat_basic_math_prompt,
    generate_cat_transfer_function_prompt,
    generate_cat_state_space_prompt,
    generate_cat_lsim_prompt,
    generate_cat_frequency_domain_prompt
]

def call_llm_api(prompt, api_choice="gemini", temperature=0.7): # Added temperature with a default
    """
    Sends a prompt to the specified LLM API and returns the response.
    """
    if api_choice == "gemini":
        try:
            import google.generativeai as genai # Import it here
            
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                print("Error: GEMINI_API_KEY environment variable not set.")
                return None
            
            genai.configure(api_key=gemini_api_key) # Now this should be found
            model = genai.GenerativeModel('gemini-2.0-flash') # Or other suitable model
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature # Use the passed temperature
            )
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # It's good practice to check if the response has parts and text
            if response.parts:
                return response.text
            else:
                # Handle cases where the response might be blocked or empty
                print("Warning: Gemini API response was empty or blocked.")
                print(f"Prompt Feedback: {response.prompt_feedback}")
                return None

        except ImportError:
            print("Error: The 'google-generativeai' library is not installed correctly.")
            return None
        except AttributeError as ae:
            # This might catch the 'genai' has no attribute 'configure' if it's due to an old version
            # or a very unusual setup after reinstall.
            print(f"AttributeError calling Gemini API (check library version or naming conflicts): {ae}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred with Gemini API: {e}")
            if "quota" in str(e).lower(): # Basic check for quota issues
                print("Quota possibly exceeded. Sleeping for a while.")
                time.sleep(60) 
            return None

    print(f"P_INFO: LLM API call with prompt:\n{prompt}\n") # For seeing the prompt
    if "MATLAB code" in prompt:
        return "figure;\nplot(1:10, rand(1,10));\ntitle('Placeholder Plot'); % Generated by placeholder API"
    elif "Describe the plot" in prompt:
        return "[plot shape: random lines, maximum: approx 1, minimum: approx 0, ...]" # Placeholder description
    return "Placeholder LLM Response"


def generate_matlab_code_prompt(category, specific_request=""):
    """Creates a prompt to ask the LLM to generate MATLAB code."""
    prompt = f"""
Please generate a concise MATLAB code snippet that produces a 2D plot.
The code should belong to the category: '{category}'.
{specific_request}

The MATLAB code should:
1. Be self-contained and runnable.
2. Include necessary variable initializations.
3. Use a standard MATLAB plotting function (e.g., plot, step, impulse, lsim).
4. Include `title()`, `xlabel()`, and `ylabel()` for clarity.
5. Add `grid on;`.
6. Ensure the plot is illustrative of the category.
7. Do NOT include any explanatory text or comments outside of the MATLAB code itself. Output only the raw MATLAB code block.

MATLAB code:
"""
    return prompt

def generate_plot_description_prompt(matlab_code):
    """Creates a prompt to ask the LLM to describe a plot from MATLAB code."""
    prompt = f"""
Given the following MATLAB code which generates a 2D plot:

```matlab
{matlab_code}
```

Describe the visual characteristics of the plot this code would produce.
Your description should be in the format:
[plot shape: <shape>, maximum: <max_val>, minimum: <min_val>, pivot points: <points>, direction: <direction>, x-intercepts: <x_int>, y-intercepts: <y_int>, final value: <val>, overshoot: <val>, rise time: <val>, settling time: <val>, other notable features: <features>]

Focus on:
- Overall shape (e.g., exponential rise, sinusoidal wave, step function, decaying oscillation).
- Approximate maximum and minimum Y-values shown.
- Key pivot points or coordinates (e.g., starts at (0,0), peak at (x,y), settles near (x,y)).
- General direction (e.g., ascending, descending, oscillating).
- Intercepts if clear.
- For control system plots (step/impulse/lsim): estimate final value, overshoot percentage, rise time, settling time if applicable and inferable.
- Any other distinct visual features.

Be concise and focus on what one would *see*. If a characteristic is not applicable or clearly inferable, you can state 'N/A' or omit it.

Description:
"""
    return prompt

def generate_dataset_samples(num_samples_to_generate, output_filename, start_index=0):
    all_samples_data = []
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r') as f:
                all_samples_data = json.load(f)
            print(f"Loaded {len(all_samples_data)} existing samples from {output_filename}")
        except json.JSONDecodeError:
            print(f"Warning: {output_filename} contains invalid JSON. Starting fresh.")
            all_samples_data = []
    
    # The category_generators list is defined above this function now.

    for i in range(num_samples_to_generate):
        current_sample_index = start_index + i
        if current_sample_index >= len(all_samples_data): # Only generate if not already present
            
            # Randomly select a category generator function
            chosen_category_generator = random.choice(category_generators)
            task_details = chosen_category_generator() # returns {"category": "...", "request_template": "..."}
            
            category = task_details["category"]
            specific_request = task_details["request_template"]
            
            print(f"\nGenerating sample {current_sample_index + 1}/{start_index + num_samples_to_generate} (Category: {category} - Request: {specific_request})")

            #Generate MATLAB code
            code_prompt = generate_matlab_code_prompt(category, specific_request)
            # Add temperature to API call for more diversity in code generation itself
            matlab_code = call_llm_api(code_prompt, temperature=0.75) # Pass temperature if your call_llm_api supports it
            
            if not matlab_code or "Placeholder LLM Response" in matlab_code :
                print(f"Failed to generate MATLAB code for sample {current_sample_index + 1}. Skipping.")
                time.sleep(2) 
                continue
            
            matlab_code = matlab_code.replace("```matlab", "").replace("```", "").strip()
            
            #Exact duplicate code check
            is_duplicate_code = False
            for existing_sample_pair in all_samples_data:
                if existing_sample_pair[0]["value"] == matlab_code:
                    is_duplicate_code = True
                    break 
            if is_duplicate_code:
                print(f"Generated MATLAB code is an exact duplicate of an existing sample. Skipping.")
                continue
            
            print(f"Generated MATLAB Code:\n{matlab_code}")

            # 2. Generate Plot Description
            desc_prompt = generate_plot_description_prompt(matlab_code)
            plot_description = call_llm_api(desc_prompt, temperature=0.7) # Slightly lower temp for description
            
            if not plot_description  or "Placeholder LLM Response" in plot_description:
                print(f"Failed to generate plot description for sample {current_sample_index + 1}. Skipping.")
                time.sleep(2)
                continue
            print(f"Generated Plot Description:\n{plot_description}")

            # 3. Format
            sample_entry = [
                {"from": "human", "value": matlab_code},
                {"from": "gpt", "value": plot_description.strip()}
            ]
            all_samples_data.append(sample_entry)

            # 4. Save periodically
            if (i + 1) % 5 == 0 or (i + 1) == num_samples_to_generate: # Save every 5 or at the end
                with open(output_filename, 'w') as f:
                    json.dump(all_samples_data, f, indent=2)
                print(f"Saved {len(all_samples_data)} samples to {output_filename}")
            
            time.sleep(6) # Adjusted for 2 calls per sample, aiming for >8s per sample cycle

        else:
            print(f"Sample {current_sample_index + 1} already exists in {output_filename}. Skipping generation.")

    # Final save
    with open(output_filename, 'w') as f:
        json.dump(all_samples_data, f, indent=2)
    print(f"Final generation complete. Total {len(all_samples_data)} samples saved to {output_filename}")


if __name__ == "__main__":

    # How many NEW samples to generate in this run?
    NUM_NEW_SAMPLES = 100
    
    #Output Filename
    #
    master_dataset_file = "matlab_plot_dataset.json" 

    current_total_samples = 0
    if os.path.exists(master_dataset_file):
        try:
            with open(master_dataset_file, 'r') as f:
                existing_data = json.load(f)
            current_total_samples = len(existing_data)
        except json.JSONDecodeError:
            pass

    print(f"Currently {current_total_samples} samples in {master_dataset_file}.")
    
    generate_dataset_samples(
        num_samples_to_generate=NUM_NEW_SAMPLES,
        output_filename=master_dataset_file,
        start_index=current_total_samples
    )
