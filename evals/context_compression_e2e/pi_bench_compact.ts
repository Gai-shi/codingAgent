function compactAndWait(ctx, options): Promise<void> {
	return new Promise((resolve, reject) => {
		ctx.compact({
			...options,
			onComplete: () => resolve(),
			onError: (error) => reject(error),
		});
	});
}

export default function (pi) {
	pi.registerCommand("bench-compact", {
		description: "Run benchmark compaction and wait for completion",
		handler: async (args, ctx) => {
			const customInstructions = args.trim() || undefined;
			await compactAndWait(ctx, { customInstructions });
		},
	});
}
