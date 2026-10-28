import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createMessage, createToolResultMessage, createUserMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { assemble } from './assemble.ts'
import { anthropicTextEvents, closeMockServers, mockServer, textEvents } from './mock-server.ts'

const contexts: Context[] = []

afterEach(async () => {
  await Promise.all(contexts.splice(0).map(ctx => ctx.fiber.dispose()))
  await closeMockServers()
  vi.unstubAllEnvs()
})

describe('Anthropic user-turn serialization', () => {
  it('coalesces tool results and following text only for Anthropic Messages', async () => {
    vi.stubEnv('ANTHROPIC_TURN_KEY', 'test-key')
    vi.stubEnv('OPENAI_TURN_KEY', 'test-key')
    const anthropic = await mockServer([{ events: anthropicTextEvents }])
    const openai = await mockServer([{ events: textEvents }])
    const ctx = new Context()
    contexts.push(ctx)
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, {
      providers: {
        'anthropic-turns': {
          apiKeyEnv: 'ANTHROPIC_TURN_KEY',
          api: 'anthropic-messages',
          baseURL: anthropic.url,
          models: [{ id: 'anthropic-model', contextWindow: 65_536, maxTokens: 4096 }],
        },
        'openai-turns': {
          apiKeyEnv: 'OPENAI_TURN_KEY',
          api: 'openai-completions',
          baseURL: openai.url,
          models: [{ id: 'openai-model', contextWindow: 65_536, maxTokens: 4096 }],
        },
      },
    })

    const callId = ToolCallId('call-1')
    const history = [
      createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'start' }] }),
      createMessage({
        role: 'assistant',
        source: { kind: 'model', provider: 'anthropic-turns', model: 'anthropic-model' },
        content: [{ type: 'tool-call', id: callId, name: 'lookup', arguments: '{}' }],
      }),
      createToolResultMessage({ callId, content: [{ type: 'text', text: 'result' }], isError: false }),
      createUserMessage({ source: { kind: 'runtime-context' }, content: [{ type: 'text', text: 'runtime context' }] }),
    ]

    await assemble(ctx, { provider: 'anthropic-turns', model: 'anthropic-model', messages: history })
    const anthropicMessages = (anthropic.requests[0] as { messages: { role: string; content: unknown }[] }).messages
    expect(anthropicMessages.map(message => message.role)).toEqual(['user', 'assistant', 'user'])
    expect(anthropicMessages[2]?.content).toMatchObject([
      { type: 'tool_result', tool_use_id: 'call-1' },
      { type: 'text', text: 'runtime context' },
    ])

    const openaiHistory = history.map(message => message)
    await assemble(ctx, { provider: 'openai-turns', model: 'openai-model', messages: openaiHistory })
    const openaiMessages = (openai.requests[0] as { messages: { role: string }[] }).messages
    expect(openaiMessages.map(message => message.role)).toEqual(['user', 'assistant', 'tool', 'user'])
  })
})
