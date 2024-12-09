#!/bin/zsh

model=$1

case $model in
    "gpt")
        echo "1: gpt-4o"
        echo "2: gpt-4o-mini"
        echo -n "Select an option: " 
        read option
        case $option in
            1)
                llm="gpt-4o"
                prefix="GPT-4o"
                ;;
            2)
                llm="gpt-4o-mini"
                prefix="GPT-4o-mini"
                ;;
            *)
                echo "Invalid option"
                exit 1
                ;;
        esac
        ;;
    "together")
        echo "1: LLaMA 3.2 90B"
        echo "2: LLaMA 3.2 11B"
        echo "3: LLaMA 3.1 70B"
        echo "4: LLaMA 3 70B"
        echo "5: Mixtral 8x22B"
        echo -n "Select an option: "
        read option
        case $option in
        1)
                llm="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
                prefix="LLaMA-3.2-90B"
                ;;
            2)
                llm="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
                prefix="LLaMA-3.2-11B"
                ;;
            3)
                llm="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
                prefix="LLaMA-3.1-70B"
                ;;
            4)
                llm="meta-llama/Meta-Llama-3-70B-Instruct-Turbo"
                prefix="LLaMA-3-70B"
                ;;
            5)
                llm="mistralai/Mixtral-8x22B-Instruct-v0.1"
                prefix="Mixtral-8x22B"
                ;;
            *)
                echo "Invalid option"
                exit 1
                ;;
        esac
        ;;
    "gemini")
        llm="gemini-1.5-flash"
        prefix="Gemini"
        ;;
    "claude")
        llm="claude-3-5-sonnet-20240620"
        prefix="Claude"
        ;;
    "grok")
        llm="grok-beta"
        prefix="Grok"
        ;;
    *)
        echo "Unknown model: $model"
        exit 1
        ;;
esac

echo ""
echo "––––––––––––––––– Using $model's $llm"
echo ""
python3 src/main.py "$model" "$llm" "$prefix"