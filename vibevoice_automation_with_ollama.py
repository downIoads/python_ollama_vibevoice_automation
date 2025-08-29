import os
import re
import subprocess
import sys
from pathlib import Path


# ------------------------------ OLLAMA STUFF ------------------------------------------------
DEFAULT_SYSTEM_PROMPT = "\nSystem Prompt Instruction: Only output raw text in your response, as it will be forwarded through a screen reader and only listened to.\n"


# clean_response enforces a certain output:
#       No duplicate sentences allowed
#       Contains only certain chars
#       Starts with 'Speaker 1:' (vibevoice requirement)
def clean_response(*, response: str) -> str:
    # Phase 1: Remove duplicate sentences
    sentences = re.findall(r'[^.?!]+[.?!]', response)
    response = " ".join(dict.fromkeys(s.strip() for s in sentences))

    # Phase 2: Remove illegal chars
    enforced_prefix = "Speaker 1: "
    response = re.sub(r'[^a-zA-Z0-9,.:()\n\- ]', '', response) # only allows a-z, A-Z, 0-9, ',', '.', '(', ')', ' ', '\n', '-'
    
    # Phase 3: Enforce prefix
    if not response.startswith(enforced_prefix):
        response = enforced_prefix + response

    return response

def persist_response(*, response: str, output_filename: str = "response.txt"):
    # write response to file
    Path(output_filename).write_text(response, encoding="utf-8")
    print(f"Response has been saved to {output_filename}.")


def query_ollama(*, user_prompt: str, target_word_amount: int, model: str = "gpt-oss:20b", system_prompt: str = DEFAULT_SYSTEM_PROMPT, iteration: int = 0, prev_response_words_amount: int = 0, output_filename: str = "response.txt"):
    # print user prompt once, but not if response has to be iteratively expanded upon
    if iteration == 0:
        print(f"Generating response for user prompt:\n\"{user_prompt}\"\n\n")

    result = subprocess.run(
        [   "ollama", 
            "run", 
            model, 
            "--hidethinking", # remove thoughts from output
        ],
        input=user_prompt+system_prompt,
        capture_output=True,
        text=True,
        check=True
    )
    response = result.stdout

    # enforce response expectation
    response = clean_response(response=response)
    
    # if current response is not longer than previous response terminate
    current_word_amount = len(response.split())
    if prev_response_words_amount > 0:
        if current_word_amount < prev_response_words_amount: # do not use <=, it often struggles to find anything new and requires multiple retries
            print(f"Aborted repeatedly trying to get a longer response because the response got shorter.")
            persist_response(response=response, output_filename="response.txt")
            return response

    # enforce length of output (if reply is shorter feed it back as another prompt to extend it)
    if current_word_amount < target_word_amount:
        print(f"Current response length is {current_word_amount} < {target_word_amount}, telling model to expand response..")

        # update the system prompt to include the previous response and a one-time notice to extend the previous response
        system_prompt_extend_reply_addition = f"\nYour response so far has been: {response}\nContinue that response to further elaborate on your reply. Ensure to include the full previous response in your extended response and do not inform the user that are you are continuing a previous response."
        if system_prompt_extend_reply_addition not in system_prompt:
            system_prompt += system_prompt_extend_reply_addition

        response = query_ollama(user_prompt=user_prompt, target_word_amount=target_word_amount, model=model, system_prompt=system_prompt, iteration=iteration+1, prev_response_words_amount=current_word_amount)

    # persist response on disk
    persist_response(response=response, output_filename="response.txt")
    
    return response


# ---------------------------------------------- VIBEVOICE STUFF ------------------------------------------------------------

DOCKER_CONTAINER_NAME = "vibevoice" # sudo docker ps -a, then check 'NAMES'


def exec_in_container(*, cmd: str):
    docker = ["sudo", "docker"]

    # ensure the named container exists and is running
    insp = subprocess.run(
        docker + ["inspect", "-f", "{{.State.Running}}", DOCKER_CONTAINER_NAME],
        capture_output=True, text=True
    )
    if insp.returncode != 0:
        raise RuntimeError(f"Container {DOCKER_CONTAINER_NAME!r} not found.")
    if insp.stdout.strip().lower() != "true":
        subprocess.run(docker + ["start", DOCKER_CONTAINER_NAME], check=True)

    # run the command inside the named container
    result = subprocess.run(
        docker + ["exec", DOCKER_CONTAINER_NAME, "bash", "-lc", cmd],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Command failed: {cmd}")
    return result.stdout


def insert_text_file(*, filepath: str):
    print("Copying text file into container..\n")

    src = os.path.expanduser(filepath)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Source file not found: {src}")

    target_dir = "/workspace/demo/text_examples"
    subprocess.run( # will overwrite existing file
        ["sudo", "docker", "cp", src, f"{DOCKER_CONTAINER_NAME}:{target_dir}/"],
        check=True
    )

    print(f"Successfully copied text file into container: {target_dir}/{Path(src).name}")


def remove_old_audio_outputs():
    print("Removing old audio files..\n")
    CMD = "rm -f /workspace/outputs/*.wav" # -f to not error if nothing is there
    
    result = exec_in_container(cmd=CMD)


def generate_audio(*, model: str, text_absolute_filepath_local: str, speaker: str):
    filename = os.path.basename(text_absolute_filepath_local)    # e.g. "1p_abs.txt" or "response.txt"
    target_path = "demo/text_examples/" + filename

    cmd = (
        "python demo/inference_from_file.py" +
        " --model_path " + model +
        " --txt_path " + target_path +
        " --speaker_names " + speaker
    )
    print(f"Generating audio using this config:\n\tModel:\t\t{model}\n\tText File:\t{target_path}\n\tSpeaker:\t{speaker}\n")

    result = exec_in_container(cmd=cmd)
    output_filename = str(Path(str(Path(filename).stem) + "_generated").with_suffix(".wav"))
    resulting_filepath = "/workspace/outputs/" + output_filename
    print(f"Successfully generated audio file: {resulting_filepath}")
    
    return resulting_filepath


def extract_file(filepath: str):
    print("Extracting file..\n")
    
    cmd = "cp " + DOCKER_CONTAINER_NAME + ":" + filepath + " ." # -f to not error if nothing is there
    dest_dir = "."

    subprocess.run(
        ["sudo", "docker", "cp", f"{DOCKER_CONTAINER_NAME}:{filepath}", dest_dir],
        check=True
    )

    cwd = os.path.abspath(dest_dir)
    print(f"Successfully extracted file to: {cwd}/{Path(filepath).name}")


def main():
    # OLLAMA CONFIG
    user_prompt = "cool facts about capybaras"
    target_word_amount = 1000
    ollama_models = ["llama3.1:8b", "gpt-oss:20b"]
    model = ollama_models[0]

    # VIBEVOICE CONFIG
    voice_model = "microsoft/VibeVoice-1.5B" # other option: "WestZhang/VibeVoice-Large-pt" but my gpu is too bad for this :(
    speaker = "en-MyOwnClonedVoice_man" # zero-shot voice cloning is super simple with vibevoice

    # ------------------------------------------------------------------------------------------------

    # before doing anything, ensure we have root (required for docker commands and freeing GPU memory)
    root_granted = (os.geteuid() == 0)
    if not root_granted:
        # request root from user
        print("This script needs root permission to run, please grant it!")
        os.execvp("sudo", ["sudo", sys.executable] + sys.argv)
        print("Thanks!\n")

    # ----------------------------------- PERFORM OLLAMA STUFF ---------------------------------------
    
    # get response (also available as response.txt in cwd)
    output_filename = "response.txt"
    query_ollama(user_prompt=user_prompt, target_word_amount=target_word_amount, model=model, output_filename=output_filename)

    # move response into vibevoice docker
    output_absolute_filepath = os.path.abspath(output_filename)
    insert_text_file(filepath=output_absolute_filepath) # copy text file to be read into container

    # ----------------------------------- PERFORM VIBEVOICE STUFF ------------------------------------

    # if the GPU has no time to free memory after doing ollama stuff it can fail to run vibevoice command (only poor people like me have this problem)
    try:
        os.system("sudo pkill -9 -f '/usr/local/bin/ollama'")
    except Exception as e:
        print(f"While trying to clear GPU memory by killing ollama I encountered this problem: {e}")

    remove_old_audio_outputs() # delete wav files in vibevoice docker that might have survived from previous runs
    
    filepath = generate_audio(
        model=voice_model,
        text_absolute_filepath_local=output_absolute_filepath,
        speaker=speaker
    )

    extract_file(filepath) # copy resulting file to cwd
    
    print("All done!\n")


main()
# What this script does
#   - takes your prompt and generates response using some ollama model
#   - takes that response and generates audio file spoken by custom voice using vibevoice
# So you basically just get an audio file that reads the response to you. This script probably does more than some multimillion dollar ai startups lul

# Assumptions:
#   A lot actually, it's not supposed to just run on your pc. It assumes docker container name for vibevoice and a certain setup, certain ollama models being installed, nvidia gpu and drivers correctly set-up etc.
