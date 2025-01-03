import unittest
from unittest.mock import patch, MagicMock
from ragxplorer.query_expansion import generate_sub_qn, generate_hypothetical_ans

class TestQueryExpansion(unittest.TestCase):

    @patch('ragxplorer.query_expansion._chat_completion')
    def test_generate_sub_qn(self, mock_chat_completion):
        mock_chat_completion.return_value = ["Sub-question 1", "Sub-question 2"]
        query = "What are the top revenue drivers for Microsoft?"
        result = generate_sub_qn(query)
        self.assertEqual(result, ["Sub-question 1", "Sub-question 2"])
        mock_chat_completion.assert_called_once()

    @patch('ragxplorer.query_expansion._chat_completion')
    def test_generate_hypothetical_ans(self, mock_chat_completion):
        mock_chat_completion.return_value = "Hypothetical answer"
        query = "What are the top revenue drivers for Microsoft?"
        result = generate_hypothetical_ans(query)
        self.assertEqual(result, "Hypothetical answer")
        mock_chat_completion.assert_called_once()

    @patch('ragxplorer.query_expansion.OpenAI')
    def test_chat_completion(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"1": "Sub-question 1", "2": "Sub-question 2"}'
        mock_client.chat.completions.create.return_value = mock_response

        from ragxplorer.query_expansion import _chat_completion
        sys_msg = "System message"
        prompt = "User prompt"
        response_format = "json_object"
        result = _chat_completion(sys_msg, prompt, response_format)
        self.assertEqual(result, ["Sub-question 1", "Sub-question 2"])
        mock_client.chat.completions.create.assert_called_once()

    @patch('ragxplorer.query_expansion.OpenAI')
    def test_chat_completion_error_handling(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        from ragxplorer.query_expansion import _chat_completion
        sys_msg = "System message"
        prompt = "User prompt"
        response_format = "json_object"
        with self.assertRaises(RuntimeError):
            _chat_completion(sys_msg, prompt, response_format)

if __name__ == '__main__':
    unittest.main()
