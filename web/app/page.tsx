"use client"

import { z } from "zod"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const formSchema = z.object({
  movieReview: z.string().min(5, {
    message: "Text to analyze must be at least 5 characters."
  })
})

export default function Home() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      movieReview: ""
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    alert(`You submitted ${JSON.stringify(values)}`);
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 text-primary-foreground">
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <FormField control={form.control} name="movieReview" render={({ field }) => {
            return <FormItem>
              <FormLabel>Message</FormLabel>
              <FormControl>
                <Input placeholder="That movie was fantastic!" {...field} />
              </FormControl>
              <FormDescription>
                Receive sentiment analysis for the provided message.
              </FormDescription>
            </FormItem>
          }} />
          <Button type="submit">Submit</Button>
        </form>
      </Form>
    </main>
  );
}
