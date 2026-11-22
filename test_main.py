import asyncio
import contextlib
import io
import os
import unittest
from unittest.mock import AsyncMock, patch

import litellm
from forecasting_tools import GeneralLlm, MonetaryCostManager

import main
from bot_helpers import check_environment


class BotTests(unittest.TestCase):
    def test_near_only_environment_and_construction(self):
        with patch.dict(os.environ, {"NEARAI_API_KEY": "dummy", "METACULUS_TOKEN": "dummy"}, clear=True):
            check_environment()
            bot = main.FallTemplateBot2026(publish_reports_to_metaculus=False)
        llms = [bot.get_llm(role, "llm") for role in ("default", "summarizer", "researcher", "parser")]
        self.assertTrue(all(llm.model == "openai/openai/gpt-6-luna" for llm in llms))
        self.assertTrue(all(llm is llms[0] for llm in llms))
        self.assertEqual(llms[0].litellm_kwargs["api_base"], "https://cloud-api.near.ai/v1")
        self.assertEqual(llms[0].litellm_kwargs["api_key"], "dummy")

    def test_unknown_model_pricing_rejected(self):
        for model in ("", "openai/unknown-model"):
            with patch.dict(os.environ, {"NEARAI_MODEL": model}), self.assertRaisesRegex(ValueError, "Unknown NEAR AI pricing"):
                main.BudgetedNearLlm()

    def test_registered_pricing(self):
        main.BudgetedNearLlm()
        cost = litellm.completion_cost(
            completion_response=litellm.ModelResponse(
                model="openai/openai/gpt-6-luna",
                usage={"prompt_tokens": 1_000, "completion_tokens": 1_000, "total_tokens": 2_000},
            ),
        )
        self.assertAlmostEqual(cost, 0.00060)

    def test_missing_key_rejected(self):
        with patch.dict(os.environ, {"METACULUS_TOKEN": "dummy"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                check_environment()

    def test_other_provider_key_without_near_rejected(self):
        with patch.dict(os.environ, {"METACULUS_TOKEN": "dummy", "OPENROUTER_API_KEY": "dummy"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                check_environment()

    def test_budget_summary_on_empty_run_and_error(self):
        for error in (None, RuntimeError("failed")):
            output = io.StringIO()
            with patch.dict(os.environ, {"MAX_LLM_COST_USD": "0.5"}), patch.object(main, "main", side_effect=error), contextlib.redirect_stdout(output):
                if error:
                    with self.assertRaises(RuntimeError):
                        main.run_with_budget()
                else:
                    main.run_with_budget()
            self.assertIn("LLM cost spent: $0.000000 (cap: $0.500000)", output.getvalue())

    def test_invalid_budget(self):
        for value in ("0", "-1", "nan", "inf", "invalid", ""):
            with patch.dict(os.environ, {"MAX_LLM_COST_USD": value}), self.assertRaises(ValueError):
                main.run_with_budget()

    def test_minibench_only_empty_dispatch(self):
        forecast = AsyncMock(return_value=[])
        with patch.object(main.FallTemplateBot2026, "forecast_on_tournament", forecast), patch.object(main, "check_environment"), patch("sys.argv", ["main.py"]), contextlib.redirect_stdout(io.StringIO()):
            main.run_with_budget()
        forecast.assert_awaited_once_with(main.MetaculusClient.CURRENT_MINIBENCH_ID, return_exceptions=True)


class CostTests(unittest.IsolatedAsyncioTestCase):
    async def test_litellm_near_cost_and_cap(self):
        llm = main.BudgetedNearLlm()
        llm.litellm_kwargs["mock_response"] = "Probability: 50%"
        llm.litellm_kwargs["api_key"] = "dummy"
        with MonetaryCostManager(0.000000001) as cm, MonetaryCostManager(), patch(
            "forecasting_tools.ai_models.general_llm.acompletion", wraps=litellm.acompletion
        ) as completion:
            answer = await llm.invoke("Give a probability")
            self.assertEqual(answer, "Probability: 50%")
            self.assertGreater(cm.current_usage, 0)
            self.assertEqual(completion.call_args.kwargs["model"], "openai/openai/gpt-6-luna")
            self.assertEqual(completion.call_args.kwargs["api_base"], "https://cloud-api.near.ai/v1")
            self.assertEqual(completion.call_args.kwargs["api_key"], "dummy")
            self.assertNotIn("tools", completion.call_args.kwargs)
            spent = cm.current_usage
            with patch.object(GeneralLlm, "_mockable_direct_call_to_model", new_callable=AsyncMock) as call:
                with self.assertRaises(RuntimeError):
                    await llm._mockable_direct_call_to_model("blocked")
                call.assert_not_awaited()
            self.assertEqual(cm.current_usage, spent)
        print(f"NEAR LiteLLM mock-response tracked cost: ${spent:.9f}; next call blocked")

    async def test_exact_cap_blocks_concurrent_calls(self):
        llm = main.BudgetedNearLlm()

        async def charged_call(prompt):
            await asyncio.sleep(0)
            MonetaryCostManager.increase_current_usage_in_parent_managers(0.5)
            return "ok"

        with MonetaryCostManager(0.5), patch.object(GeneralLlm, "_mockable_direct_call_to_model", side_effect=charged_call) as call:
            results = await asyncio.gather(*(llm._mockable_direct_call_to_model("test") for _ in range(3)), return_exceptions=True)
        self.assertEqual(results[0], "ok")
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results[1:]))
        self.assertEqual(call.await_count, 1)


if __name__ == "__main__":
    unittest.main()
